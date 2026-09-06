#!/usr/bin/env python3
r"""generate_real_qa.py — the single real_qa data-generation pipeline.

Reproduces `data/real_qa/<model>/` (the directly usable memorization
dataset: 100 tasks = 10 datasets × 10 subtasks × 50 rows, test == train) from
scratch. One file, five subcommands:

  load     HF download + normalize the 10 QA datasets  -> data/raw/<ds>/<split>.jsonl       (CPU)
  filter   rejection-sample contamination-free rows by sampling a served base
           model 5x/question and dropping any it can already answer
                                                        -> data/completions/<model>/...      (GPU; needs a vLLM OpenAI endpoint)
  export   drop model-known rows (+ write an audit)     -> data/cleaned/<model>/<ds>/{split,split.known}.jsonl  (CPU)
  build    cleaned -> 10×10×50 tasks, test == train     -> data/real_qa/<model>/             (CPU)
  run      serve a base model on vLLM, then filter+export for it (orchestrates)              (GPU)

Layout (everything under data_generation/real_qa/data/ is ignored scratch;
the final dataset is written below the repository's data/real_qa directory):

  data_generation/real_qa/
    configs/{datasets,models}.yaml      # dataset sources + filter targets; model list + sampling params
    data/raw/<ds>/<split>.jsonl         # load   output
    data/completions/<model>/...        # filter output (all samples + contains verdict)
    data/cleaned/<model>/<ds>/...       # export output (kept + .known audit)
  data/real_qa/<model>/task_t/...       # build output (t = 0..99)

Policy: CONTAINS-ONLY (no LLM judge). A row is "known" (dropped) iff the base
model produced the gold as a word-boundary substring in any of its 5 samples.
No separate model judge is used: ``known`` is derived from the substring test.

Examples:
  python generate_real_qa.py load
  python generate_real_qa.py run    --model_name qwen3_4b_base --hf_id Qwen/Qwen3-4B-Base --tp 2
  python generate_real_qa.py build  --model qwen3_4b_base             # -> data/real_qa/qwen3_4b_base/ (100×50)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import re
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent          # data_generation/real_qa
REPO = HERE.parent.parent                        # repo root
SCRATCH = HERE / "data"                          # gitignored scratch (raw/completions/cleaned)
CONFIGS = HERE / "configs"
COMMITTED = REPO / "data" / "real_qa"

# Training source order is alphabetical and deterministic.
DEFAULT_ORDER = [
    "arc_challenge", "arc_easy", "medmcqa", "nq_open", "openbookqa",
    "popqa", "sciq", "squad", "triviaqa", "webquestions",
]


# ===========================================================================
# shared helpers
# ===========================================================================

def _dedup(xs: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for s in xs:
        if not s:
            continue
        k = s.lower().strip()
        if k in seen:
            continue
        seen.add(k); out.append(s)
    return out


def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(p) if l.strip()]


# ===========================================================================
# load: HF download + normalize  (former 01_load_and_normalize.py)
# ===========================================================================

def _norm_triviaqa(row):
    a = row.get("answer") or {}
    primary = (a.get("value") or "").strip()
    aliases = [s.strip() for s in (a.get("aliases") or []) if s and s.strip()]
    norm_al = [s.strip() for s in (a.get("normalized_aliases") or []) if s and s.strip()]
    return {"question": (row.get("question") or "").strip(),
            "gold_answers": _dedup([primary, *aliases, *norm_al]),
            "meta": {"qid_src": row.get("question_id")}}


def _norm_nq_open(row):
    answers = row.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]
    return {"question": (row.get("question") or "").strip(),
            "gold_answers": _dedup([s.strip() for s in answers if isinstance(s, str) and s.strip()]),
            "meta": {}}


def _norm_popqa(row):
    primary = (row.get("obj") or "").strip()
    aliases: List[str] = []
    pa = row.get("possible_answers")
    if isinstance(pa, str):
        try:
            aliases = json.loads(pa)
        except Exception:
            aliases = [pa]
    elif isinstance(pa, list):
        aliases = pa
    return {"question": (row.get("question") or "").strip(),
            "gold_answers": _dedup([primary, *(s.strip() for s in aliases if isinstance(s, str) and s.strip())]),
            "meta": {"subj": row.get("subj"), "prop": row.get("prop"),
                     "s_pop": row.get("s_pop"), "o_pop": row.get("o_pop")}}


def _norm_squad(row):
    a = row.get("answers") or {}
    texts = a.get("text") or []
    return {"question": (row.get("question") or "").strip(),
            "gold_answers": _dedup([s.strip() for s in texts if isinstance(s, str) and s.strip()]),
            "meta": {"title": row.get("title")}}


def _norm_webquestions(row):
    answers = row.get("answers") or []
    return {"question": (row.get("question") or "").strip(),
            "gold_answers": _dedup([s.strip() for s in answers if isinstance(s, str) and s.strip()]),
            "meta": {"url": row.get("url")}}


def _mcqa_pick(label_or_key, choices_text, choices_label=None) -> str:
    """Resolve gold text from an answerKey/cop label and the choices arrays."""
    if choices_label is not None:
        try:
            idx = list(choices_label).index(label_or_key)
            return str(choices_text[idx]).strip()
        except Exception:
            pass
    if isinstance(label_or_key, str) and len(label_or_key) == 1:
        idx = ord(label_or_key.upper()) - ord("A")
        if 0 <= idx < len(choices_text):
            return str(choices_text[idx]).strip()
    if isinstance(label_or_key, int) and 0 <= label_or_key < len(choices_text):
        return str(choices_text[label_or_key]).strip()
    return ""


def _norm_openbookqa(row):
    ch = row.get("choices") or {}
    gold = _mcqa_pick(row.get("answerKey"), ch.get("text") or [], ch.get("label") or [])
    return {"question": (row.get("question_stem") or "").strip(),
            "gold_answers": [gold] if gold else [],
            "meta": {"id": row.get("id"), "fact1": row.get("fact1"),
                     "humanScore": row.get("humanScore")}}


def _norm_sciq(row):
    return {"question": (row.get("question") or "").strip(),
            "gold_answers": [s.strip() for s in [row.get("correct_answer") or ""] if s.strip()],
            "meta": {"support": row.get("support")}}


def _norm_arc(row):
    ch = row.get("choices") or {}
    gold = _mcqa_pick(row.get("answerKey"), ch.get("text") or [], ch.get("label") or [])
    return {"question": (row.get("question") or "").strip(),
            "gold_answers": [gold] if gold else [],
            "meta": {"id": row.get("id")}}


def _norm_medmcqa(row):
    opts = [row.get(k) or "" for k in ("opa", "opb", "opc", "opd")]
    cop = row.get("cop")
    gold = _mcqa_pick(cop, opts) if cop is not None else ""
    return {"question": (row.get("question") or "").strip(),
            "gold_answers": [gold.strip()] if gold and gold.strip() else [],
            "meta": {"id": row.get("id"), "subject_name": row.get("subject_name"),
                     "topic_name": row.get("topic_name")}}


NORMALIZERS = {
    "triviaqa": _norm_triviaqa, "nq_open": _norm_nq_open, "popqa": _norm_popqa,
    "squad": _norm_squad, "webquestions": _norm_webquestions,
    "openbookqa": _norm_openbookqa, "sciq": _norm_sciq,
    "arc_challenge": _norm_arc, "arc_easy": _norm_arc,
    "medmcqa": _norm_medmcqa,
}


def _stable_qid(dataset: str, split: str, idx: int, question: str) -> str:
    h = hashlib.sha1(f"{dataset}::{split}::{idx}::{question}".encode("utf-8")).hexdigest()[:10]
    return f"{dataset}::{split}::{h}"


def _write_split(d, split_role, ds, out_root: Path, cap: int) -> int:
    norm = NORMALIZERS[d["name"]]
    out_dir = out_root / d["name"]; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_role}.jsonl"
    n_in = n_out = 0
    with open(out_path, "w") as f:
        for idx, row in enumerate(ds):
            n_in += 1
            try:
                rec = norm(dict(row))
            except Exception as e:
                print(f"    !! norm err idx={idx}: {e}", file=sys.stderr); continue
            q = rec.get("question") or ""
            golds = rec.get("gold_answers") or []
            if not q or not golds:
                continue
            out = {"qid": _stable_qid(d["name"], split_role, idx, q),
                   "dataset": d["name"], "split": split_role, "type": d["type"],
                   "question": q, "gold_answers": golds, "meta": rec.get("meta") or {}}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
            if cap and n_out >= cap:
                break
    print(f"    -> {out_path}  in={n_in}  out={n_out}")
    return n_out


def cmd_load(args):
    import yaml
    import datasets as hfds
    cfg = yaml.safe_load(open(args.config))
    out_root = Path(args.out_dir).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    print(f"out_root = {out_root}")
    selected = args.datasets or [d["name"] for d in cfg["datasets"]]
    manifest = {}
    for d in cfg["datasets"]:
        if d["name"] not in selected:
            continue
        print(f"\n== {d['name']} ({d['type']}) ==")
        manifest[d["name"]] = {}
        for role in args.splits:
            if args.max_per_split >= 0:
                cap = args.max_per_split
            else:
                override = (d.get("cap") or {}).get(role)
                cap = int(override if override is not None
                          else (cfg.get("max_per_split") or {}).get(role) or 0)
            sname = d["splits"].get(role)
            if not sname:
                print(f"  (no {role} split configured)"); continue
            try:
                print(f"  load {d['hf_path']}  config={d.get('hf_config')}  split={sname}")
                ds = hfds.load_dataset(d["hf_path"], d.get("hf_config"), split=sname)
            except Exception as e:
                print(f"  !! load failed: {e}", file=sys.stderr)
                manifest[d["name"]][role] = {"error": str(e)}; continue
            n = _write_split(d, role, ds, out_root, cap)
            manifest[d["name"]][role] = {"n": n, "cap": cap}
    json.dump(manifest, open(out_root / "manifest.json", "w"), indent=2)
    print(f"\nmanifest -> {out_root/'manifest.json'}")


# ===========================================================================
# filter: rejection-sample contamination-free rows  (former 02_run_completions.py)
# ===========================================================================

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in set(string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def contains_any(pred: str, golds: List[str]) -> bool:
    """Word-boundary substring match (prevents short aliases like 'BR' matching
    'broom'). Tradeoff: gold '50s' won't match '1950s' — but the longer alias is
    almost always present too."""
    p = _norm_text(pred)
    if not p:
        return False
    for g in golds:
        gn = _norm_text(g)
        if gn and re.search(rf"(?<!\w){re.escape(gn)}(?!\w)", p):
            return True
    return False


def _resolve_target(cfg, ds_name, role, cli_override):
    if cli_override is not None and cli_override >= 0:
        return cli_override or None
    for d in cfg.get("datasets") or []:
        if d.get("name") == ds_name and "target" in d:
            return (d["target"] or {}).get(role)
    return (cfg.get("target_per_split") or {}).get(role)


async def _rejection_sample(client, model, recs, out_path: Path, target, sp,
                            max_concurrent, seed, log_every=200):
    """Stream recs through the model with an in-flight window; stop submitting
    once kept >= target. Resumable via the existing output file."""
    import asyncio
    done_qids, kept_count, n_done = set(), 0, 0
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                done_qids.add(r["qid"]); n_done += 1
                if not r.get("any_contains"):
                    kept_count += 1
            except Exception:
                pass
    target_int = target if target is not None and target > 0 else math.inf
    if kept_count >= target_int:
        print(f"  already at target ({kept_count}/{target_int}), skip")
        return {"processed": n_done, "kept": kept_count, "target": target, "pool_size": len(recs)}

    rng = random.Random(seed)
    order = list(range(len(recs))); rng.shuffle(order)
    pending = [recs[i] for i in order if recs[i]["qid"] not in done_qids]

    async def run_one(rec):
        try:
            r = await client.completions.create(
                model=model, prompt=f"Question: {rec['question']}\nAnswer:",
                n=sp["n"], temperature=sp["temperature"], top_p=sp["top_p"],
                max_tokens=sp["max_tokens"], stop=sp["stop"])
            samples = [ch.text.strip() for ch in r.choices]
        except Exception as e:
            samples = [""] * sp["n"]
            print(f"  err {rec['qid']}: {e}", file=sys.stderr)
        cps = [contains_any(s, rec["gold_answers"]) for s in samples]
        return {**{k: rec[k] for k in ("qid", "dataset", "split", "type", "question", "gold_answers")},
                "model": model, "samples": samples, "contains_pass": cps,
                "any_contains": any(cps), "needs_judge": not any(cps)}

    f = open(out_path, "a")
    it = iter(pending); inflight = set(); stop = False

    def fill():
        if stop:
            return
        while len(inflight) < max_concurrent:
            try:
                rec = next(it)
            except StopIteration:
                return
            inflight.add(asyncio.create_task(run_one(rec)))

    fill()
    while inflight:
        done, _ = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
        for dt in done:
            inflight.discard(dt)
            r = dt.result()
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n_done += 1
            if not r.get("any_contains"):
                kept_count += 1
            if n_done % log_every == 0 or kept_count == target_int:
                f.flush()
                pct = "" if target_int == math.inf else f" ({100*kept_count/target_int:.0f}%)"
                print(f"  processed={n_done}  kept={kept_count}/{target}{pct}", flush=True)
        if kept_count >= target_int:
            stop = True
        if not stop:
            fill()
    f.flush(); f.close()
    print(f"  DONE  processed={n_done}  kept={kept_count}  target={target}  pool={len(recs)}")
    return {"processed": n_done, "kept": kept_count, "target": target, "pool_size": len(recs)}


def cmd_filter(args):
    import asyncio
    import yaml
    from openai import AsyncOpenAI

    async def _run():
        cfg = yaml.safe_load(open(args.config))
        sp = {"n": args.n, "temperature": args.temperature, "top_p": args.top_p,
              "max_tokens": args.max_tokens, "stop": args.stop}
        client = AsyncOpenAI(base_url=args.endpoint, api_key="EMPTY", timeout=180.0)
        model = args.model_id
        if not model:
            ml = await client.models.list(); model = ml.data[0].id
        print(f"endpoint={args.endpoint}  served_model={model}  model_name={args.model_name}")
        in_root = Path(args.in_dir); out_root = Path(args.out_dir) / args.model_name
        cli_targets = {"train": args.target_train, "eval": args.target_eval}
        plan = []
        for ds_dir in sorted(in_root.iterdir()):
            if not ds_dir.is_dir() or (args.datasets and ds_dir.name not in args.datasets):
                continue
            for split in args.splits:
                p = ds_dir / f"{split}.jsonl"
                if p.exists():
                    plan.append((ds_dir.name, split, p,
                                 _resolve_target(cfg, ds_dir.name, split, cli_targets.get(split))))
        for ds_name, split, in_path, tgt in plan:
            recs = load_jsonl(in_path)
            out_dir = out_root / ds_name; out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{split}.jsonl"
            print(f"\n== {ds_name}/{split}  pool={len(recs)}  target={tgt}  -> {out_path}")
            if args.overwrite and out_path.exists():
                out_path.unlink()
            await _rejection_sample(client, model, recs, out_path, tgt, sp,
                                    max_concurrent=args.max_concurrent, seed=args.seed)
    asyncio.run(_run())


# ===========================================================================
# export: drop model-known rows  (former 04_filter_and_export.py)
# ===========================================================================

def _export_clean(in_path: Path, kept_path: Path, dropped_path: Path) -> Dict[str, int]:
    """Contains-only mode: a record is `known` iff any_contains is True."""
    n = kept = drop = 0
    fk = open(kept_path, "w"); fd = open(dropped_path, "w")
    for rec in load_jsonl(in_path):
        n += 1
        carry = {k: rec[k] for k in ("qid", "dataset", "split", "type", "question", "gold_answers")
                 if k in rec}
        carry["meta"] = rec.get("meta") or {}
        if bool(rec.get("any_contains")):
            fd.write(json.dumps({**carry, "drop_reason": "contains",
                                 "samples": rec.get("samples"), "judge_results": None},
                                ensure_ascii=False) + "\n")
            drop += 1
        else:
            fk.write(json.dumps(carry, ensure_ascii=False) + "\n"); kept += 1
    fk.close(); fd.close()
    return {"n": n, "kept": kept, "dropped": drop}


def cmd_export(args):
    in_root = Path(args.in_dir) / args.model_name
    out_root = Path(args.out_dir) / args.model_name
    if not in_root.exists():
        sys.exit(f"no completions dir at {in_root} (run `filter` first)")
    for ds_dir in sorted(in_root.iterdir()):
        if not ds_dir.is_dir() or (args.datasets and ds_dir.name not in args.datasets):
            continue
        for split in args.splits:
            in_path = ds_dir / f"{split}.jsonl"
            if not in_path.exists():
                continue
            out_dir = out_root / ds_dir.name; out_dir.mkdir(parents=True, exist_ok=True)
            stats = _export_clean(in_path, out_dir / f"{split}.jsonl", out_dir / f"{split}.known.jsonl")
            pct = 100.0 * stats["dropped"] / max(1, stats["n"])
            print(f"  {ds_dir.name}/{split}: kept {stats['kept']:>6}/{stats['n']:>6}  drop={pct:>5.1f}%")


# ===========================================================================
# build: cleaned -> uniform N/task, test == train
# ===========================================================================

def _items_from_records(recs):
    out = []
    for r in recs:
        out.append({"id": r["qid"], "query": r["question"],
                    "answer": r["gold_answers"][0] if r["gold_answers"] else "",
                    "answer_aliases": r["gold_answers"][1:],
                    "dataset": r["dataset"], "type": r["type"]})
    return out


def _texts_from_records(recs):
    out = []
    for r in recs:
        a = r["gold_answers"][0] if r["gold_answers"] else ""
        q = r["question"]
        if q and a:
            out.append(f"Question: {q}\nAnswer: {a}")
    return out


def cmd_build(args):
    """cleaned -> committed data/real_qa/<model>/: each dataset (fixed order) is
    split into `--subtasks` consecutive CL tasks of `--n` rows, with test ==
    train (pure memorization; the held-out eval split is dropped). Default
    10 datasets × 10 subtasks × 50 rows = 100 tasks.

    Per dataset we draw `n * subtasks` rows with ONE `random.sample` call (same
    RNG consumption as the old one-task-per-dataset build) and slice them into
    `subtasks` chunks — so `--subtasks 1 --n 500` reproduces the old layout and
    `--subtasks 10 --n 50` is exactly that same data re-chunked. Bit-reproduces
    the committed artifact given the same cleaned/, order, subtasks, N and seed."""
    src = Path(args.cleaned_root) / args.model
    if not src.exists():
        sys.exit(f"no cleaned dir: {src} (run `filter`+`export` first)")
    out = Path(args.out_root) / args.model
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)  # global RNG, consumed sequentially across datasets
    per_ds = args.n * args.subtasks           # rows drawn per dataset
    tasks_seen, counts, ds_used = [], [], []
    gt = 0                                     # running global task index
    for ds in args.task_order:
        train_p = src / ds / "train.jsonl"
        if not train_p.exists():
            print(f"  skip dataset {ds}: no train.jsonl at {train_p}")
            continue
        recs = load_jsonl(train_p)
        texts = _texts_from_records(recs)
        items = _items_from_records(recs)
        take = min(per_ds, len(texts))
        if len(texts) < per_ds:
            print(f"  ⚠ {ds}: only {len(texts)} rows (< {per_ds} = {args.n}×{args.subtasks}) "
                  f"— produces fewer/short subtasks")
        idx = sorted(random.sample(range(len(texts)), take))
        ds_texts = [texts[i] for i in idx]
        ds_items = [items[i] for i in idx]
        ds_used.append(ds)
        for c in range(args.subtasks):         # slice into consecutive subtasks
            chunk_texts = ds_texts[c * args.n:(c + 1) * args.n]
            chunk_items = ds_items[c * args.n:(c + 1) * args.n]
            if not chunk_texts:
                break                          # ran out of rows for this dataset
            td = out / f"task_{gt}"; td.mkdir(parents=True, exist_ok=True)
            json.dump(chunk_texts, open(td / "train_texts.json", "w"), ensure_ascii=False)
            json.dump(chunk_items, open(td / "train_items.json", "w"), ensure_ascii=False)
            json.dump(chunk_items, open(td / "test_items.json", "w"), ensure_ascii=False)  # test == train
            tasks_seen.append({"index": gt, "name": ds, "subtask": c,
                               "n_train": len(chunk_texts), "n_eval": len(chunk_texts)})
            counts.append(len(chunk_texts))
            print(f"  task {gt:>3} = {ds:<15} subtask {c:>2}  n={len(chunk_texts)}")
            gt += 1
    manifest = {"dataset_type": "cl_qa", "model_name": args.model,
                "n_tasks": len(tasks_seen), "n_datasets": len(ds_used),
                "subtasks_per_dataset": args.subtasks, "uniform_n": args.n,
                "test_equals_train": True, "subsample_seed": args.seed,
                "source": args.model, "datasets_in_order": ds_used,
                "tasks_in_order": tasks_seen}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    uniform = len(set(counts)) <= 1
    print(f"\n{len(tasks_seen)} tasks ({len(ds_used)} datasets × {args.subtasks} subtasks) -> {out}"
          f"   (per-task min={min(counts)} max={max(counts)} uniform={uniform})")


# ===========================================================================
# run: serve a base model on vLLM, then filter + export  (former run_one_model.sh)
# ===========================================================================

def cmd_run(args):
    import yaml
    sp = (yaml.safe_load(open(args.config)) or {}).get("sampling") or {}
    vllm = os.environ.get("VLLM", "vllm")
    cuda = ",".join(str(i) for i in range(args.tp))
    logdir = SCRATCH.parent / "logs"; logdir.mkdir(parents=True, exist_ok=True)
    serve_log = open(logdir / f"serve_{args.model_name}.log", "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda,
           "GLOO_SOCKET_IFNAME": "lo", "NCCL_SOCKET_IFNAME": "lo"}
    print(f"serve {args.hf_id} on CUDA {cuda} -> :{args.port}")
    proc = subprocess.Popen(
        [vllm, "serve", args.hf_id, "--port", str(args.port), "--host", "127.0.0.1",
         "--served-model-name", args.model_name, "--tensor-parallel-size", str(args.tp),
         "--gpu-memory-utilization", "0.80", "--max-model-len", "4096"],
        stdout=serve_log, stderr=subprocess.STDOUT, env=env)
    try:
        import urllib.request
        endpoint = f"http://127.0.0.1:{args.port}/v1"
        up = False
        for _ in range(150):
            if proc.poll() is not None:
                sys.exit(f"vLLM serve died early — see {logdir}/serve_{args.model_name}.log")
            try:
                if b'"id"' in urllib.request.urlopen(endpoint + "/models", timeout=5).read():
                    up = True; break
            except Exception:
                pass
            time.sleep(10)
        if not up:
            sys.exit("serve did not come up in time")
        print("serve up; filtering...")
        fa = argparse.Namespace(
            endpoint=endpoint, model_id=None, model_name=args.model_name, config=args.config,
            in_dir=str(SCRATCH / "raw"), out_dir=str(SCRATCH / "completions"),
            datasets=args.datasets, splits=args.splits, target_train=None, target_eval=None,
            max_concurrent=args.max_concurrent, n=sp.get("n", 5),
            temperature=sp.get("temperature", 0.7), top_p=sp.get("top_p", 0.95),
            max_tokens=sp.get("max_tokens", 64), stop=sp.get("stop", ["\n", "Question:", "Q:"]),
            seed=args.seed, overwrite=args.overwrite)
        cmd_filter(fa)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    print("export...")
    cmd_export(argparse.Namespace(
        model_name=args.model_name, in_dir=str(SCRATCH / "completions"),
        out_dir=str(SCRATCH / "cleaned"), datasets=args.datasets, splits=args.splits))
    print(f"=== DONE {args.model_name} ===")


# ===========================================================================
# CLI
# ===========================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Single real_qa data-generation pipeline (load/filter/export/build/run).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    dsc = str(CONFIGS / "datasets.yaml")

    pl = sub.add_parser("load", help="HF download + normalize -> data/raw/")
    pl.add_argument("--config", default=dsc)
    pl.add_argument("--out_dir", default=str(SCRATCH / "raw"))
    pl.add_argument("--datasets", nargs="*", default=None)
    pl.add_argument("--splits", nargs="*", default=["train", "eval"])
    pl.add_argument("--max_per_split", type=int, default=-1)
    pl.set_defaults(func=cmd_load)

    pf = sub.add_parser("filter", help="rejection-sample via a served base model -> data/completions/")
    pf.add_argument("--endpoint", required=True, help="vLLM OpenAI endpoint, e.g. http://127.0.0.1:8890/v1")
    pf.add_argument("--model_id", default=None, help="served id (auto-detect if omitted)")
    pf.add_argument("--model_name", required=True)
    pf.add_argument("--config", default=dsc)
    pf.add_argument("--in_dir", default=str(SCRATCH / "raw"))
    pf.add_argument("--out_dir", default=str(SCRATCH / "completions"))
    pf.add_argument("--datasets", nargs="*", default=None)
    pf.add_argument("--splits", nargs="*", default=["train", "eval"])
    pf.add_argument("--target_train", type=int, default=None)
    pf.add_argument("--target_eval", type=int, default=None)
    pf.add_argument("--max_concurrent", type=int, default=64)
    pf.add_argument("--n", type=int, default=5)
    pf.add_argument("--temperature", type=float, default=0.7)
    pf.add_argument("--top_p", type=float, default=0.95)
    pf.add_argument("--max_tokens", type=int, default=64)
    pf.add_argument("--stop", nargs="*", default=["\n", "Question:", "Q:"])
    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--overwrite", action="store_true")
    pf.set_defaults(func=cmd_filter)

    pe = sub.add_parser("export", help="drop known rows -> data/cleaned/")
    pe.add_argument("--model_name", required=True)
    pe.add_argument("--in_dir", default=str(SCRATCH / "completions"))
    pe.add_argument("--out_dir", default=str(SCRATCH / "cleaned"))
    pe.add_argument("--datasets", nargs="*", default=None)
    pe.add_argument("--splits", nargs="*", default=["train", "eval"])
    pe.set_defaults(func=cmd_export)

    pb = sub.add_parser("build", help="cleaned -> committed data/real_qa/<model>/ (100 tasks of 50, test==train)")
    pb.add_argument("--model", required=True, help="e.g. qwen3_4b_base")
    pb.add_argument("--n", type=int, default=50, help="rows per (sub)task")
    pb.add_argument("--subtasks", type=int, default=10,
                    help="consecutive CL tasks per dataset (10 datasets × 10 × n=50 = 100 tasks)")
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--cleaned_root", default=str(SCRATCH / "cleaned"))
    pb.add_argument("--out_root", default=str(COMMITTED))
    pb.add_argument("--task_order", nargs="*", default=DEFAULT_ORDER)
    pb.set_defaults(func=cmd_build)

    pr = sub.add_parser("run", help="serve a base model + filter + export (one model, end-to-end)")
    pr.add_argument("--model_name", required=True)
    pr.add_argument("--hf_id", required=True)
    pr.add_argument("--tp", type=int, default=1)
    pr.add_argument("--port", type=int, default=8890)
    pr.add_argument("--config", default=dsc)
    pr.add_argument("--datasets", nargs="*", default=None)
    pr.add_argument("--splits", nargs="*", default=["train", "eval"])
    pr.add_argument("--max_concurrent", type=int, default=64)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--overwrite", action="store_true")
    pr.set_defaults(func=cmd_run)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
