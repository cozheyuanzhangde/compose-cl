#!/usr/bin/env python3
"""Task-level Successive Halving (TSH) over continual-learning horizons.

A configuration is trained and evaluated at progressively longer task horizons.
At each rung, configurations are ranked by final retention averaged over all
configured training seeds, and only the leading configurations continue.

Two-phase cutting (topk_rung defaults to the LAST rung -> percentage-cut all the
way down the ladder, then ONE top-k cut into the deepest, most expensive run):
  • budget <  topk_rung : percentage cut — keep the top keep_fracs[k] FRACTION of
    the configs still ALIVE. keep_fracs is a LIST, one LOCAL fraction per percentage
    cut in rung order (cut k uses keep_fracs[k]); it cuts the CURRENT cohort, not the
    original pool, so 0.5 halves the survivors each rung.
  • budget >= topk_rung : keep only the top `topk` configs (a hard COUNT) — the
    single final cut into the deepest, most expensive rung.

Features
  • Multi-GPU: one worker thread per GPU pulls configs off a shared queue.
  • Resume: a config's results.json at a rung is reused if present, so a killed
    run restarts and continues from where it stopped (survivors are recomputed
    deterministically from the cached scores).

Config (YAML) — see ``tsh/configs/symbol_qa.yaml``. Two candidate schemas:
'categories' (cross-category method combinations x per-method hp_sets) or the flat
'methods' list ({name, flags, grid}, cartesian product of grid -> CLI flags).

Usage:
  PY=<train/eval python> python tsh/run.py CONFIG.yaml --gpus 0,1,2,3
  python tsh/run.py CONFIG.yaml --dry_run
"""
import argparse, itertools, json, math, os, queue, signal, subprocess, sys, threading

PY = os.environ.get("PY", sys.executable)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ranking objective is HARDCODED, not a config knob: always rank/cut by 'final'
# (mean of the last accuracy-matrix row). Ranking by 'forget' rewards non-learners
# (a config that learns nothing forgets nothing) and 'diag' ignores retention, so
# neither is a valid selector.
RANK_METRIC = "final"


# ----------------------------------------------------------------- config / grid
def load_cfg(path):
    import yaml
    cfg = yaml.safe_load(open(path))
    # Each cell trains and evaluates every seed; cuts use mean final retention.
    _seeds = cfg.get("seeds", cfg.get("seed", 42))
    if not isinstance(_seeds, (list, tuple)):
        _seeds = [_seeds]
    cfg["seeds"] = [int(s) for s in _seeds]
    if not cfg["seeds"]:
        raise SystemExit("config 'seeds' must be a non-empty list")
    cfg.setdefault("boxed", True)
    cfg.setdefault("common_flags", "")
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("grad_accum", 1)
    cfg.setdefault("gen_batch_size", 256)
    if cfg.get("metric", RANK_METRIC) != RANK_METRIC:
        raise SystemExit(f"metric is hardcoded to {RANK_METRIC!r} and cannot be set in configs "
                         f"(got metric={cfg['metric']!r}); remove it from the YAML.")
    cfg["metric"] = RANK_METRIC                     # ranking objective is not configurable
    cfg.setdefault("exclude_reg_merge", False)     # drop reg×merge combos? (default: keep)
    if "rungs" not in cfg:
        raise SystemExit("config must define 'rungs'")
    # Default = the LAST rung: percentage-cut all the way down the ladder, then a
    # single top-k cut into the deepest (most expensive) run. Setting topk_rung
    # below the last rung makes every rung from there on run the same top-k set —
    # a flat tail that prunes nothing across the deep rungs (main() warns).
    cfg.setdefault("topk_rung", cfg["rungs"][-1])  # budget at which we switch to top-k
    cfg.setdefault("topk", 10)                     # final hard count kept at/after topk_rung
    # keep_fracs: a LIST of LOCAL keep-fractions, ONE per percentage cut (the cuts
    # whose next rung is < topk_rung), in rung order. Cut k keeps ceil(frac_k * ALIVE)
    # of the configs still ALIVE — a local share of the current cohort, NOT the
    # original pool. The final cut into topk_rung uses the hard `topk` count instead.
    rungs = cfg["rungs"]
    n_pct_cuts = sum(1 for i in range(len(rungs) - 1) if rungs[i + 1] < cfg["topk_rung"])
    kf = cfg.get("keep_fracs")
    if kf is None:
        raise SystemExit("config must define 'keep_fracs' (a list, one fraction per "
                         f"percentage cut; this ladder has {n_pct_cuts})")
    kf = [float(x) for x in (kf if isinstance(kf, (list, tuple)) else [kf] * n_pct_cuts)]
    if len(kf) != n_pct_cuts:
        raise SystemExit(f"keep_fracs has {len(kf)} item(s) but this ladder has "
                         f"{n_pct_cuts} percentage cut(s) (rungs={rungs}, "
                         f"topk_rung={cfg['topk_rung']}); give exactly one per cut.")
    cfg["keep_fracs"] = kf
    ct = cfg.get("cell_timeout_min")                # {rung: minutes} or None; kills hung cells
    cfg["cell_timeout_min"] = ({int(k): float(v) for k, v in ct.items()} if ct else None)
    # Dataset-level separation: everything lives under results/tsh/<dataset>/ so runs on
    # different datasets (symbol_qa, llm_qa, ...) never share cells or records. Derived
    # from the data_dir basename by default; override with an explicit `dataset:` field.
    cfg["dataset_dir"] = cfg.get("dataset") or os.path.basename(
        str(cfg.get("data_dir", "")).rstrip("/")) or "misc"
    cfg["root"] = os.path.join("results", "tsh", cfg["dataset_dir"], cfg["run_name"])
    # CELL STORE vs RECORDS. A cell's train+eval (<tag>/rung_<b>/) is determined by
    # the method+HP and the base training HPs — NOT by the funnel schedule. So two
    # configs that differ only in keep_fracs/topk compute the SAME cells and can
    # SHARE a store, reusing each other's checkpoints/results. Set the same
    # `cell_store: <name>` in both -> cells live at results/tsh/<name>/<tag>/rung_<b>/.
    # Defaults to run_name (so existing configs are unchanged). Funnel records
    # (trace/leaderboard/progress) ALWAYS stay per-run under cfg["root"]. Safety: each
    # cell writes a cell.json HP-manifest; preflight_reuse() refuses to reuse a cell
    # whose recorded HPs differ from this config's (guards a mis-shared store).
    cfg["cell_store"] = cfg.get("cell_store") or cfg["run_name"]
    cfg["cell_root"] = os.path.join("results", "tsh", cfg["dataset_dir"], cfg["cell_store"])
    return cfg


def _abbr(v):
    s = str(v).replace(".", "p").replace("-", "m")
    return s


def _hp_to_flags_bits(hp):
    """One hyperparameter dict -> ("--k v --k2 v2", ["k<abbr>", ...])."""
    parts, bits = [], []
    for k, v in hp.items():
        parts.append(f"--{k} {v}")
        bits.append(f"{k}{_abbr(v)}")
    return " ".join(parts), bits


def _method_variants(m):
    """A method entry {name, flags, hp_sets} -> [(seg_tag, flags), ...], one per
    hp_set. hp_sets is an explicit LIST of HP dicts (NOT cartesian-expanded — each
    entry is one fully-specified setting, so a multi-HP method is a list of tuples);
    <=3 by convention. A method with no hp_sets yields a single no-HP variant."""
    base = (m.get("flags") or "").strip()
    out = []
    for hp in (m.get("hp_sets") or [{}]):
        hp_flags, bits = _hp_to_flags_bits(hp)
        seg = m["name"] + ("_" + "_".join(bits) if bits else "")
        out.append((seg, " ".join(p for p in (base, hp_flags) if p).strip()))
    return out


def expand_categories(cfg):
    """Cross-CATEGORY combinations: choose AT MOST ONE method per category (or
    none), then cross every chosen method's hp_set variants. Combination happens
    ONLY across categories (no two methods from the same category co-occur), so the
    pool size is

        N0 = prod over categories of (1 + sum over methods of #variants)

    The all-off pick is the 'vanilla' baseline. Backward-compatible alternative to
    the flat methods+grid expand() (a config uses one schema or the other)."""
    per_cat = []
    for _cat, methods in cfg["categories"].items():
        opts = [None]                                  # None = this category off
        for m in methods:
            opts.extend(_method_variants(m))
        per_cat.append(opts)
    cands, dropped = [], 0
    exclude_reg_merge = cfg.get("exclude_reg_merge", False)
    for combo in itertools.product(*per_cat):
        chosen = [o for o in combo if o is not None]
        if chosen:
            tag = "+".join(seg for seg, _ in chosen)
            flags = " ".join(f for _, f in chosen).strip()
        else:
            tag, flags = "vanilla", ""                 # all categories off
        if exclude_reg_merge and not _combo_ok(flags): # optionally drop reg×merge combos
            dropped += 1
            continue
        cands.append({"tag": tag, "flags": flags})
    if dropped:
        print(f"[tsh] excluded {dropped} reg×merge combos (exclude_reg_merge=true; "
              f"EWC/SI on an O-LoRA/LoRA-merge path is weak)", flush=True)
    seen = {}                                          # de-dup by tag
    for c in cands:
        seen[c["tag"]] = c
    return list(seen.values())


def _combo_ok(flags):
    """Identify reg×merge combos: EWC/SI together with O-LoRA / LoRA-merging. These
    RUN now (train.py only warns, and the regularizers are wired into the O-LoRA
    loop), but folding/recreating adapters each task breaks EWC/SI's fixed
    parameter-identity assumption, so reg is empirically useless-to-harmful there.
    expand_categories drops them only when exclude_reg_merge=true; by default every
    cross-category pairing is kept. Returns False for a reg×merge combo."""
    f = f" {flags} "
    reg = (" --SI " in f) or (" --online_ewc " in f)
    merge = (" --olora " in f) or (" --merge_lora_per_task " in f)
    return not (reg and merge)


def expand(cfg):
    """Build the candidate pool. Two schemas (use one):
      • categories: cross-category combinations x per-method hp_sets — pool =
        product of (1 + #variants) per category (see expand_categories).
      • methods:    flat list, each {name, flags, grid} cartesian-expanded."""
    if cfg.get("categories"):
        return expand_categories(cfg)
    cands = []
    for m in cfg["methods"]:
        base, grid = m.get("flags", ""), (m.get("grid") or {})
        keys = list(grid)
        for combo in (itertools.product(*[grid[k] for k in keys]) if keys else [()]):
            parts, bits = [base], []
            for k, v in zip(keys, combo):
                parts.append(f"--{k} {v}")
                bits.append(f"{k}{_abbr(v)}")
            tag = m["name"] + ("__" + "_".join(bits) if bits else "")
            cands.append({"tag": tag, "flags": " ".join(p for p in parts if p).strip()})
    # de-dup by tag
    seen = {}
    for c in cands:
        seen[c["tag"]] = c
    return list(seen.values())


# ----------------------------------------------------------------- run one cell
def rung_dir(cfg, tag, budget):
    return os.path.join(cfg["cell_root"], tag, f"rung_{budget}")        # SHARED cell store


def results_json(cfg, tag, budget, seed):
    return os.path.join(rung_dir(cfg, tag, budget), "eval", f"seed_{seed}", "results.json")


def cell_manifest_path(cfg, tag, budget):
    return os.path.join(rung_dir(cfg, tag, budget), "cell.json")


def cell_signature(cfg, flags, budget):
    """Everything that determines a cell's train+eval RESULT (NOT the funnel
    schedule). Recorded in cell.json and checked before reuse: two configs sharing a
    cell_store must agree on this signature, else reusing the cell would be wrong."""
    return {
        "model": cfg["model"], "data_dir": cfg["data_dir"], "n_tasks": budget,
        "lr": str(cfg["lr"]), "epochs": cfg["epochs"], "batch_size": cfg["batch_size"],
        "grad_accum": cfg["grad_accum"], "max_seq_len": cfg["max_seq_len"],
        "seeds": cfg["seeds"], "boxed": bool(cfg["boxed"]),
        "common_flags": " ".join(cfg["common_flags"].split()),
        "max_new_tokens": cfg["max_new_tokens"], "gen_batch_size": cfg["gen_batch_size"],
        "method_flags": " ".join((flags or "").split()),
        "prefix_resume": bool(cfg.get("prefix_resume", False)),   # final-only+continued cells are a distinct reuse class
    }


def write_cell_manifest(cfg, cand, budget):
    """Record the cell's HP signature next to its outputs (<cell>/cell.json) so a
    later config sharing the store can verify it's reusing an identical computation."""
    p = cell_manifest_path(cfg, cand["tag"], budget)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"tag": cand["tag"], "budget": budget, "first_run": cfg["run_name"],
                   "signature": cell_signature(cfg, cand["flags"], budget)}, f, indent=2)


def score_of(path):
    """The ranking score from a results.json: RANK_METRIC='final' = mean of the
    last accuracy-matrix row. Higher is better. Robust to a 1-row matrix."""
    try:
        m = json.load(open(path))["accuracy_matrix"]
    except Exception:
        return None
    if not m:
        return None
    last = m[-1]                                    # 'final' = mean of last row
    return 100.0 * sum(last) / len(last)


def build_cmds(cfg, cand, budget, seed):
    out = rung_dir(cfg, cand["tag"], budget)
    boxed = "--boxed_answers" if cfg["boxed"] else ""
    common = f"{cfg['common_flags']} {boxed}".strip()
    # --replay_token must be forwarded to eval too, or recall collapses.
    eval_rt = "--replay_token" if " --replay_token " in f" {cand['flags']} " else ""
    # --task_order_seed must be forwarded to eval too, or the shuffled-order accuracy
    # matrix reads each column from the WRONG on-disk task (same class of bug as
    # --replay_token above). It lives in common_flags (-> train.py); pull it out for evaluate.py.
    _cf = cfg["common_flags"].split()
    eval_tos = (f"--task_order_seed {_cf[_cf.index('--task_order_seed') + 1]}"
                if "--task_order_seed" in _cf else "")
    # --prune_checkpoints (evaluate.py): after eval, keep ONLY the last per-task checkpoint
    # (OSRM writes a full model/task -> unbounded disk otherwise). Default on; a side
    # effect, so NOT part of cell_signature (doesn't affect results / reuse identity).
    prune = "--prune_checkpoints" if cfg.get("prune_checkpoints", True) else ""
    # Prefix-resume (opt-in via cfg['prefix_resume']): a promoted config CONTINUES from
    # the previous rung's final checkpoint + method/RNG state instead of retraining the
    # task prefix from base. Requires the prior rung trained with --keep_resume (state
    # survives completion) and its last checkpoint kept (prune keeps after_task_{N-1}).
    # Eval becomes --final_only: only the final model spans all N tasks, and the search
    # ranks solely on 'final'. Cuts BOTH the redundant prefix training and the B^2 eval.
    rprefix = keep = fdo = ""
    if cfg.get("prefix_resume"):
        rungs = cfg["rungs"]
        i = rungs.index(budget) if budget in rungs else 0
        if i > 0:                                    # not the first rung -> resume from the previous one
            prev = rungs[i - 1]
            rprefix = f"--resume_from {rung_dir(cfg, cand['tag'], prev)}/ck --start_task {prev}"
        if i < len(rungs) - 1:                       # not the last rung -> keep state for the next rung
            keep = "--keep_resume"
        fdo = "--final_only"
    train = (f"{PY} train.py --model {cfg['model']} --data_dir {cfg['data_dir']} "
             f"--output_dir {out}/ck --n_tasks {budget} --epochs {cfg['epochs']} "
             f"--batch_size {cfg['batch_size']} --grad_accum {cfg['grad_accum']} "
             f"--lr {cfg['lr']} --max_seq_len {cfg['max_seq_len']} --seed {seed} "
             f"{common} {cand['flags']} {rprefix} {keep}")
    # NOTE: --seed IS forwarded to evaluate.py (it keys ck/eval seed-subdirs on it).
    evaluation = (f"{PY} evaluate.py --model {cfg['model']} --data_dir {cfg['data_dir']} "
                  f"--checkpoint_dir {out}/ck --output_dir {out}/eval --n_tasks {budget} "
                  f"--max_new_tokens {cfg['max_new_tokens']} "
                  f"--gen_batch_size {cfg['gen_batch_size']} --seed {seed} "
                  f"{boxed} {eval_rt} {eval_tos} {prune} {fdo}")
    return out, train, evaluation


def _cell_timeout_s(cfg, budget):
    """Per-rung wall-clock cap (seconds) for one train/eval step, or None."""
    ct = cfg.get("cell_timeout_min")
    return float(ct[budget]) * 60.0 if (ct and budget in ct) else None


def _run_step(cmd, env, logpath, timeout_s):
    """Run one shell step, streaming to logpath. Returns (returncode|None, timed_out).
    Robust kill: own session + SIGKILL the whole process group on timeout, so a hung
    train.py (e.g. runaway replay generation) and its children are reaped."""
    with open(logpath, "w") as f:
        p = subprocess.Popen(cmd, shell=True, cwd=REPO, env=env, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=timeout_s), False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            return None, True


class SeedFailed(Exception):
    """A seed's train/eval failed (crash / timeout / no output). Raised to HALT the
    whole sweep so the user can inspect — we never rank on an unequal set of seeds."""


def run_one(cfg, cand, budget, gpu, log):
    """Train+eval one config at one budget on one GPU across ALL cfg['seeds']; the
    cell score is the MEAN 'final' over ALL seeds (aggregate-then-cut). Resume-safe
    PER SEED (a present per-seed results.json is reused).
    FAIL-HARD: any seed failure raises SeedFailed to halt the sweep — every candidate
    is ranked on the same N seeds, or the run stops for inspection (no partial means)."""
    tag = cand["tag"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false"}
    tmo = _cell_timeout_s(cfg, budget)
    seed_scores = []
    for seed in cfg["seeds"]:
        rj = results_json(cfg, tag, budget, seed)
        if not os.path.isfile(rj):                     # not computed yet -> run this seed
            out, train, evaluation = build_cmds(cfg, cand, budget, seed)
            os.makedirs(os.path.join(out, "_logs"), exist_ok=True)
            rc, to = _run_step(train, env, os.path.join(out, "_logs", f"train_seed{seed}.log"), tmo)
            if to or rc != 0:
                why = f"TIMEOUT >{int((tmo or 0)/60)}min" if to else f"rc={rc}"
                raise SeedFailed(f"{tag}@{budget} seed{seed}: TRAIN {why} "
                                 f"(see {out}/_logs/train_seed{seed}.log)")
            rc, to = _run_step(evaluation, env, os.path.join(out, "_logs", f"eval_seed{seed}.log"), tmo)
            if to or rc != 0 or not os.path.isfile(rj):
                why = "TIMEOUT" if to else (f"rc={rc}" if rc != 0 else "no results.json")
                raise SeedFailed(f"{tag}@{budget} seed{seed}: EVALUATION {why} "
                                 f"(see {out}/_logs/eval_seed{seed}.log)")
        sc = score_of(rj)
        if sc is None:
            raise SeedFailed(f"{tag}@{budget} seed{seed}: results.json has no usable "
                             f"accuracy_matrix ({rj})")
        seed_scores.append(sc)
    write_cell_manifest(cfg, cand, budget)             # record HPs (incl. seeds list) for reuse
    mean = sum(seed_scores) / len(seed_scores)
    log(f"    [{tag}@{budget}] seeds "
        + " ".join(f"{sd}:{sc:.1f}" for sd, sc in zip(cfg["seeds"], seed_scores))
        + f"  -> mean {mean:.1f}")
    return mean


def run_rung(cfg, survivors, budget, gpus, log):
    """Run all survivors at `budget` across the GPU pool. Returns {tag: score}.
    FAIL-HARD: if any seed-run fails, stop launching new cells, let in-flight cells
    finish, then HALT the whole sweep (SystemExit) — nothing is cut on an incomplete
    seed set. Finished seed-cells are reused, so a fixed re-run resumes where it left off."""
    q = queue.Queue()
    for c in survivors:
        q.put(c)
    scores, lock, done = {}, threading.Lock(), [0]
    abort, err = threading.Event(), [None]

    def worker(gpu):
        while not abort.is_set():                      # stop pulling new cells once a failure is seen
            try:
                c = q.get_nowait()
            except queue.Empty:
                return
            try:
                s = run_one(cfg, c, budget, gpu, log)
            except SeedFailed as e:
                with lock:
                    if err[0] is None:
                        err[0] = e
                abort.set()
                log(f"    !! SEED FAILURE — halting (letting in-flight cells finish): {e}")
                return
            with lock:
                scores[c["tag"]] = s
                done[0] += 1
                k = done[0]
            log(f"    [{k}/{len(survivors)}] {c['tag']}@{budget}: {s:.1f}")

    ts = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if err[0] is not None:
        raise SystemExit(
            f"\n[tsh] SEED FAILURE at rung {budget} — sweep HALTED for inspection (nothing was cut).\n"
            f"  {err[0]}\n"
            f"  Likely an OOM / bad GPU node / hang. Check that log, fix it, then re-run the SAME\n"
            f"  command: finished seed-cells are reused, so it resumes where it stopped.\n")
    return scores


def survivors_after(cfg, cut_idx, n_alive):
    """How many configs survive the `cut_idx`-th percentage cut: a LOCAL fraction
    keep_fracs[cut_idx] of the configs CURRENTLY ALIVE (not the original pool), so a
    fraction of 0.5 halves the live cohort each time. Floored at 1, capped at
    n_alive. (Percentage cuts are numbered 0,1,... in rung order; the final cut into
    topk_rung is the hard `topk` count, handled in select().)"""
    frac = cfg["keep_fracs"][cut_idx]
    return max(1, min(math.ceil(n_alive * frac), n_alive))


def select(cfg, ranked, cut_idx, next_budget):
    """Survivors for the NEXT rung. Next rung below topk_rung -> LOCAL percentage cut
    keeping keep_fracs[cut_idx] of the live cohort. Next rung at/past topk_rung ->
    the hard `topk` count only (so the rung AT topk_rung already trains just topk)."""
    if next_budget < cfg["topk_rung"]:
        return ranked[:survivors_after(cfg, cut_idx, len(ranked))]
    return ranked[:min(cfg["topk"], len(ranked))]


# ----------------------------------------------------------------- leaderboard
def write_leaderboard(cfg, cands, history):
    rungs = cfg["rungs"]
    path = os.path.join(cfg["root"], "leaderboard.md")
    os.makedirs(cfg["root"], exist_ok=True)
    lines = [f"# TSH — {cfg['run_name']}", "",
             f"_metric={cfg['metric']} (mean over seeds={cfg['seeds']})  rungs={rungs}  "
             f"keep_fracs={cfg['keep_fracs']} (local)  "
             f"topk_rung={cfg['topk_rung']}  topk={cfg['topk']}_", "",
             "| config | " + " | ".join(f"r{b}" for b in rungs) + " |",
             "|" + "---|" * (len(rungs) + 1)]
    # sort by deepest available score
    def deepest(tag):
        h = history.get(tag, {})
        for b in reversed(rungs):
            if h.get(b) is not None:
                return (rungs.index(b), h[b])
        return (-1, -1e9)
    for c in sorted(cands, key=lambda c: deepest(c["tag"]), reverse=True):
        h = history.get(c["tag"], {})
        cells = [(f"{h[b]:.1f}" if h.get(b) is not None else ("x" if b in h else "·")) for b in rungs]
        lines.append(f"| {c['tag']} | " + " | ".join(cells) + " |")
    open(path, "w").write("\n".join(lines) + "\n")


# -------------------------------------------------------- analysis trace + HTML
# A structured record of the WHOLE search — every config that ran at each rung, its
# score/rank, and whether it SURVIVED or was CUT — so later analysis/visualization
# (e.g. an HTML funnel) can show exactly what was eliminated along the way.
def _rung_event(budget, ran, scores, survived_tags, cut_type, keep_count, note):
    """One rung's record: ranking (by score), survivors, eliminated, failures."""
    ok = sorted([c for c in ran if scores.get(c["tag"]) is not None],
                key=lambda c: scores[c["tag"]], reverse=True)
    failed = [c["tag"] for c in ran if scores.get(c["tag"]) is None]
    ranking = [{"tag": c["tag"], "flags": c["flags"], "score": round(scores[c["tag"]], 3),
                "rank": r + 1, "survived": c["tag"] in survived_tags}
               for r, c in enumerate(ok)]
    surv_scores = [scores[c["tag"]] for c in ok if c["tag"] in survived_tags]
    return {
        "budget": budget, "n_ran": len(ran), "cut_type": cut_type,
        "keep_count": keep_count, "note": note,
        "cut_score": round(min(surv_scores), 3) if surv_scores else None,
        "n_survived": sum(1 for c in ok if c["tag"] in survived_tags),
        "n_eliminated": sum(1 for c in ok if c["tag"] not in survived_tags) + len(failed),
        "survived": [c["tag"] for c in ok if c["tag"] in survived_tags],
        "eliminated": [c["tag"] for c in ok if c["tag"] not in survived_tags] + failed,
        "failed": failed, "ranking": ranking,
    }


def write_trace(cfg, cands, rung_events, winner, history):
    """Dump the full search trace to results/tsh/<run>/trace.json (rewritten each
    rung, so it reflects progress and survives resume). `history` is the full
    {tag: {rung: score}} matrix (score null = failed cell) so the per-config
    trajectory is reconstructable straight from the trace, not only the per-rung
    rankings."""
    trace = {
        "run_name": cfg["run_name"], "dataset": cfg["data_dir"], "model": cfg["model"],
        "metric": cfg["metric"], "rungs": cfg["rungs"], "topk_rung": cfg["topk_rung"],
        "topk": cfg["topk"], "keep_fracs": cfg["keep_fracs"],
        "n0": len(cands),
        "candidates": [{"tag": c["tag"], "flags": c["flags"]} for c in cands],
        "history": {t: {str(b): s for b, s in h.items()} for t, h in history.items()},
        "rung_events": rung_events,
        "completed_rungs": [e["budget"] for e in rung_events],
        "winner": winner,
    }
    os.makedirs(cfg["root"], exist_ok=True)
    with open(os.path.join(cfg["root"], "trace.json"), "w") as f:
        json.dump(trace, f, indent=2)
    return trace


def write_progress_html(cfg, trace):
    """Self-contained static HTML funnel from the trace: per rung, every config that
    ran, ranked by score, badged SURVIVED / cut / FAILED. No JS or deps."""
    import html as _h
    esc = _h.escape
    css = ("body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222}"
           "h1{margin:0 0 4px}.meta{color:#666;font-size:13px;margin-bottom:14px}"
           ".winner{background:#fffbe6;border:1px solid #f0d000;padding:6px 10px;border-radius:6px;"
           "display:inline-block;margin-bottom:16px}"
           ".funnel{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}"
           ".fcell{border:1px solid #ddd;border-radius:6px;padding:8px 12px;font-size:13px}"
           ".fcell b{font-size:18px}table{border-collapse:collapse;width:100%;margin:6px 0 22px;font-size:13px}"
           "th,td{border-bottom:1px solid #eee;padding:4px 8px;text-align:left}th{background:#fafafa}"
           "td.s{font-variant-numeric:tabular-nums;text-align:right;width:64px}"
           "td.r{color:#999;text-align:right;width:42px}"
           ".b{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}"
           ".surv{background:#e3f6e3;color:#137a13}.cut{background:#fde8e8;color:#b22}"
           ".fail{background:#fff0d9;color:#a60}tr.cutrow{opacity:.7}tr.winrow{background:#fffbe6}"
           "h2{margin:22px 0 6px;font-size:16px}code{font-size:11px;color:#555}")
    rungs = trace["rungs"]
    h = [f"<!doctype html><meta charset=utf-8><title>TSH — {esc(trace['run_name'])}</title>",
         f"<style>{css}</style>", f"<h1>TSH · {esc(trace['run_name'])}</h1>",
         f"<div class=meta>dataset {esc(str(trace['dataset']))} · model {esc(str(trace['model']))} · "
         f"metric <b>{esc(trace['metric'])}</b> · rungs {esc(str(rungs))} · N0={trace['n0']} · "
         f"topk={trace['topk']}@{trace['topk_rung']} · completed {esc(str(trace['completed_rungs']))}</div>"]
    if trace.get("winner"):
        h.append(f"<div class=winner>&#127942; winner: <b>{esc(trace['winner'])}</b></div>")
    h.append("<div class=funnel>")
    for e in trace["rung_events"]:
        kept = e["keep_count"] if e["cut_type"] != "final" else e["n_survived"]
        h.append(f"<div class=fcell>rung {e['budget']}<br><b>{e['n_ran']}</b> ran<br>&rarr; {kept} kept"
                 f"<br><span style=color:#999>{esc(e['note'] or e['cut_type'])}</span></div>")
    h.append("</div>")

    # config × rung trajectory: the whole funnel in ONE matrix — every config's score
    # at each rung, where it dropped out (red edge), and the winner highlighted.
    hist, last_rung, status = {}, {}, {}
    for e in trace["rung_events"]:
        b = e["budget"]
        for row in e["ranking"]:
            hist.setdefault(row["tag"], {})[b] = row["score"]
            last_rung[row["tag"]] = b
            status[row["tag"]] = "survived" if row["survived"] else "cut"
        for t in e["failed"]:
            hist.setdefault(t, {})[b] = None
            last_rung[t] = b
            status[t] = "failed"
    winner_tag = trace.get("winner")
    def _key(t):
        lr = last_rung.get(t, -1)
        sc = hist.get(t, {}).get(lr)
        return (rungs.index(lr) if lr in rungs else -1, sc if sc is not None else -1e9)
    h.append("<h2>trajectory <span style='color:#999;font-weight:400;font-size:13px'>— score by "
             "config &times; rung; where each config dropped out, winner highlighted</span></h2>")
    h.append("<table><tr><th>config</th>"
             + "".join(f"<th class=s>r{b}</th>" for b in rungs) + "<th>outcome</th></tr>")
    for t in sorted(hist, key=_key, reverse=True):
        st = status.get(t)
        if t == winner_tag:
            outcome, rowcls = "<span class='b surv'>WINNER &#127942;</span>", "winrow"
        elif st == "survived":
            outcome, rowcls = "<span class='b surv'>finalist</span>", "survrow"
        elif st == "failed":
            outcome, rowcls = f"<span class='b fail'>FAILED @ r{last_rung[t]}</span>", "cutrow"
        else:
            outcome, rowcls = f"<span class='b cut'>dropped after r{last_rung[t]}</span>", "cutrow"
        cells = []
        for b in rungs:
            row_h = hist.get(t, {})
            if b not in row_h:
                cells.append("<td class=s style='color:#ccc'>&middot;</td>")
            elif row_h[b] is None:
                cells.append("<td class=s><span class='b fail'>&times;</span></td>")
            else:
                sc = row_h[b]
                a = max(0.0, min(1.0, sc / 100.0)) * 0.4
                drop = (b == last_rung.get(t)) and st == "cut"
                style = f"background:rgba(19,122,19,{a:.2f})" + (";border-right:3px solid #b22" if drop else "")
                cells.append(f"<td class=s style='{style}'>{sc:.1f}</td>")
        h.append(f"<tr class={rowcls}><td>{esc(t)}</td>{''.join(cells)}<td>{outcome}</td></tr>")
    h.append("</table>")

    h.append("<h2>per-rung detail</h2>")
    for e in trace["rung_events"]:
        cl = f", cut line {e['cut_score']:.1f}" if e.get("cut_score") is not None else ""
        h.append(f"<h3>rung {e['budget']} — {e['n_ran']} ran, {e['n_survived']} survived{cl} "
                 f"<span style='color:#999;font-weight:400'>({esc(e['note'] or e['cut_type'])})</span></h3>")
        h.append("<table><tr><th>#</th><th>score</th><th>config</th><th>status</th><th>flags</th></tr>")
        for row in e["ranking"]:
            badge = ("<span class='b surv'>SURVIVED</span>" if row["survived"]
                     else "<span class='b cut'>cut</span>")
            h.append(f"<tr class={'survrow' if row['survived'] else 'cutrow'}>"
                     f"<td class=r>{row['rank']}</td><td class=s>{row['score']:.1f}</td>"
                     f"<td>{esc(row['tag'])}</td><td>{badge}</td>"
                     f"<td><code>{esc(row['flags'])}</code></td></tr>")
        for t in e["failed"]:
            h.append(f"<tr class=cutrow><td class=r>&mdash;</td><td class=s>&mdash;</td>"
                     f"<td>{esc(t)}</td><td><span class='b fail'>FAILED</span></td><td></td></tr>")
        h.append("</table>")
    os.makedirs(cfg["root"], exist_ok=True)
    open(os.path.join(cfg["root"], "progress.html"), "w").write("\n".join(h))


# ----------------------------------------------------------------- reuse preflight
def preflight_reuse(cfg, cands, log):
    """Guard + report the (possibly SHARED) cell store before the search runs. Any
    cell already holding results.json must carry a cell.json manifest whose HP
    signature matches THIS config; otherwise reusing it would silently corrupt the
    run, so abort with guidance. Legacy cells (results.json but no manifest) get one
    backfilled. Single-threaded and cheap (path stats + small JSON reads)."""
    present = reused = backfilled = 0
    for c in cands:
        for b in cfg["rungs"]:
            if not os.path.isfile(results_json(cfg, c["tag"], b, cfg["seeds"][0])):
                continue                                # first seed present => cell computed; verify sig
            present += 1
            cur = cell_signature(cfg, c["flags"], b)
            mp = cell_manifest_path(cfg, c["tag"], b)
            if os.path.isfile(mp):
                try:
                    old = json.load(open(mp)).get("signature", {})
                except Exception:
                    old = None
                if old != cur:
                    diffs = sorted(k for k in cur if not old or old.get(k) != cur.get(k))
                    raise SystemExit(
                        f"\n[tsh] CELL-STORE COLLISION — refusing to reuse a mismatched cell.\n"
                        f"  cell:       {rung_dir(cfg, c['tag'], b)}\n"
                        f"  cell_store: {cfg['cell_store']!r}   this config: {cfg['run_name']!r}\n"
                        f"  The cell on disk was computed with DIFFERENT hyperparameters\n"
                        f"  (mismatched: {diffs}). Reusing it would corrupt results.\n"
                        f"  Fix: give this config its own `cell_store:` name, or align the HPs.\n")
                reused += 1
            else:
                write_cell_manifest(cfg, c, b)         # legacy/first-touch: record HPs now
                backfilled += 1
    if present:
        log(f"[tsh] cell_store '{cfg['cell_store']}': {present} finished cell(s) on disk "
            f"({reused} HP-verified, {backfilled} backfilled) -> reused, not recomputed.")
    else:
        log(f"[tsh] cell_store '{cfg['cell_store']}': no finished cells yet -> computing fresh.")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids")
    ap.add_argument("--per_gpu", type=int, default=1,
                    help="cells to run concurrently PER GPU (shared-VRAM packing; "
                         "e.g. 2 on a 141GB H200). Total workers = len(gpus)*per_gpu.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    gpus = [int(g) for g in str(args.gpus).split(",") if g != ""]
    per_gpu = max(1, args.per_gpu)
    work_gpus = [g for g in gpus for _ in range(per_gpu)]   # 2/gpu -> [0,1,..,0,1,..]
    cands = expand(cfg)
    if cfg["topk_rung"] < cfg["rungs"][-1]:
        print(f"[warn] topk_rung={cfg['topk_rung']} < last rung {cfg['rungs'][-1]}: "
              f"every rung >= {cfg['topk_rung']} runs the same top-{cfg['topk']} set "
              f"(flat tail — the deepest rung prunes nothing). Set "
              f"topk_rung={cfg['rungs'][-1]} for a single top-k cut at the deepest rung.",
              flush=True)

    def log(m):
        print(m, flush=True)

    if args.dry_run:
        print(f"# TSH dry-run — {cfg['run_name']}")
        print(f"metric={cfg['metric']}  rungs={cfg['rungs']}  topk_rung={cfg['topk_rung']}  "
              f"topk={cfg['topk']}  gpus={gpus}")
        print(f"seeds={cfg['seeds']}  ->  each cell runs all {len(cfg['seeds'])} seed(s); "
              f"score = MEAN 'final' across seeds, then cut")
        print(f"keep_fracs={cfg['keep_fracs']} (local: fraction of the live cohort per cut)")
        print(f"\n{len(cands)} candidate configs:")
        for c in cands:
            print(f"  {c['tag']:<34} {c['flags']}")
        n = len(cands)
        print("\nprojected funnel (deterministic, by count):")
        for i, b in enumerate(cfg["rungs"]):
            last = i == len(cfg["rungs"]) - 1
            if not last and cfg["rungs"][i + 1] < cfg["topk_rung"]:
                note = f"  -> keep {cfg['keep_fracs'][i] * 100:.0f}% local"
            elif not last:
                note = f"  -> top-{cfg['topk']} cut"
            else:
                note = ""
            tail = " (top-k tail)" if b >= cfg["topk_rung"] else ""
            print(f"  rung {b:>4}: {n} configs{tail}{note}")
            if not last:
                nb = cfg["rungs"][i + 1]
                n = (survivors_after(cfg, i, n) if nb < cfg["topk_rung"]
                     else min(cfg["topk"], n))
        return

    preflight_reuse(cfg, cands, log)                  # share/reuse cells across funnels (HP-checked)
    log(f"=== TSH {cfg['run_name']}  gpus={gpus} x{per_gpu}/gpu = {len(work_gpus)} workers  "
        f"{len(cands)} configs ===")
    history, survivors, ranked, rung_events = {}, cands, cands, []
    for i, budget in enumerate(cfg["rungs"]):
        log(f"\n=== RUNG {i+1}/{len(cfg['rungs'])}  budget={budget} tasks  ({len(survivors)} configs) ===")
        scores = run_rung(cfg, survivors, budget, work_gpus, log)
        for tag, s in scores.items():
            history.setdefault(tag, {})[budget] = s
        ranked = sorted([c for c in survivors if scores.get(c["tag"]) is not None],
                        key=lambda c: scores[c["tag"]], reverse=True)
        log(f"  rung {budget} ranking:")
        for c in ranked:
            log(f"    {scores[c['tag']]:6.1f}  {c['tag']}")
        write_leaderboard(cfg, cands, history)
        ran_here = survivors                                  # configs that ran at this rung
        last = i == len(cfg["rungs"]) - 1
        if last:
            survived = {c["tag"] for c in ranked}            # all finalists "survive" to the end
            cut_type, keep_count, note = "final", len(ranked), ""
        else:
            next_b = cfg["rungs"][i + 1]
            survivors = select(cfg, ranked, i, next_b)
            survived = {c["tag"] for c in survivors}
            if next_b < cfg["topk_rung"]:
                cut_type, keep_count = "percentage", len(survivors)
                note = f"keep {cfg['keep_fracs'][i] * 100:.0f}% local"
            else:
                cut_type, keep_count, note = "topk", len(survivors), f"top-{cfg['topk']} cut"
        rung_events.append(_rung_event(budget, ran_here, scores, survived, cut_type, keep_count, note))
        winner = ranked[0]["tag"] if (last and ranked) else None
        try:                                                 # viz output must never kill the sweep
            write_progress_html(cfg, write_trace(cfg, cands, rung_events, winner, history))
        except Exception as e:
            log(f"  [warn] trace/html write failed: {e!r}")
        if last:
            break
        log(f"  -> {len(survivors)} survive to rung {cfg['rungs'][i+1]}: "
            + ", ".join(c["tag"] for c in survivors))

    if ranked:
        log(f"\n=== WINNER: {ranked[0]['tag']} ({cfg['metric']}={history[ranked[0]['tag']][cfg['rungs'][-1]]:.1f}) ===")
    log(f"leaderboard -> {os.path.join(cfg['root'], 'leaderboard.md')}")
    log(f"trace/html  -> {os.path.join(cfg['root'], 'trace.json')} , progress.html")
    log(f"cells       -> {cfg['cell_root']}/<tag>/rung_<b>/  (shared store '{cfg['cell_store']}')")


if __name__ == "__main__":
    main()
