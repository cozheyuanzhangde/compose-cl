#!/usr/bin/env python
"""Aggregate paper metrics across seeds from ``evaluate.py`` result matrices.

Writes ``metrics_summary.json`` and a compact Markdown table beneath
``results/final/<dataset>/``. Per-cell metric caches avoid repeatedly loading
large generation logs.
"""
import argparse
import glob
import json
import os
import statistics

def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _peak_mem(mem_log_path):
    """Peak GPU (max_reserved_mb) and host (rss_peak_mb) from a mem_log.jsonl."""
    gpu = host = None
    if not os.path.exists(mem_log_path):
        return gpu, host
    try:
        with open(mem_log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                g, h = d.get("max_reserved_mb"), d.get("rss_peak_mb")
                if g is not None:
                    gpu = g if gpu is None else max(gpu, g)
                if h is not None:
                    host = h if host is None else max(host, h)
    except OSError:
        pass
    return gpu, host


def compute_cell_metrics(d, ck_seed_dir, wk, horizons):
    """All per-seed metrics from a loaded results.json dict (+ its ck mem_log)."""
    M = d.get("accuracy_matrix") or []
    n_ckpt = len(M)
    n_task = len(M[0]) if M else 0
    full = (n_ckpt == n_task and n_ckpt > 1)          # full N×N vs sparse (--final_only)
    last = M[-1] if M else []

    out = {
        "n_ckpt": n_ckpt, "n_task": n_task, "full_matrix": full,
        "final_acc": _mean(last),
        "avg_final_accuracy_file": d.get("avg_final_accuracy"),   # cross-check
        "diag": d.get("avg_immediate_accuracy") if full else None,
        "forget_bwt": d.get("avg_forgetting") if full else None,
        "fwt": d.get("avg_forward_transfer") if full else None,
        "train_time_sec": d.get("total_train_time_sec"),
    }
    for k in wk:
        out[f"W{k}"] = _mean(last[-k:]) if last else None

    # box-emission: fraction of gens emitting a parseable \boxed{}. Average over the
    # FINAL model's rows (after_task == n-1) and over the diagonal (each task as learned).
    dl = d.get("detail_log", [])
    fin = n_task - 1
    be_f = [e["stats"]["box_emission"] for e in dl
            if e.get("after_task") == fin and (e.get("stats") or {}).get("box_emission") is not None]
    be_d = [e["stats"]["box_emission"] for e in dl
            if e.get("after_task") == e.get("eval_task") and (e.get("stats") or {}).get("box_emission") is not None]
    out["box_emission_final"] = _mean(be_f)
    out["box_emission_diag"] = _mean(be_d)

    # horizon profile (only meaningful on the full matrix): final & W10 after task h-1,
    # restricted to the h tasks seen by then — the keeps-recent-vs-all curve vs horizon.
    prof = {}
    if full:
        for h in horizons:
            if 1 <= h <= n_ckpt:
                row = M[h - 1]
                seen = row[:h]
                prof[str(h)] = {"final": _mean(seen), "W10": _mean(seen[-10:])}
    out["horizon_profile"] = prof

    gpu, host = _peak_mem(os.path.join(ck_seed_dir, "mem_log.jsonl"))
    out["peak_gpu_mb"], out["peak_host_mb"] = gpu, host
    return out


def cell_metrics_cached(results_path, ck_seed_dir, wk, horizons, use_cache):
    """Compute and cache one cell's aggregate metrics."""
    cell_dir = os.path.dirname(results_path)
    cache = os.path.join(cell_dir, "_metrics.json")
    rp_mtime = os.path.getmtime(results_path)
    metrics_fresh = (use_cache and os.path.exists(cache)
                     and os.path.getmtime(cache) >= rp_mtime)
    if metrics_fresh:
        try:
            with open(cache) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    with open(results_path) as f:           # the one heavy read
        d = json.load(f)
    m = compute_cell_metrics(d, ck_seed_dir, wk, horizons)
    try:
        with open(cache, "w") as f:
            json.dump(m, f, indent=2)
    except OSError:
        pass
    return m


def aggregate(per_seed, wk):
    """mean / sample-std / n over seeds for every numeric metric key."""
    keys = ["final_acc", "diag", "forget_bwt", "fwt",
            "box_emission_final", "box_emission_diag",
            "train_time_sec", "peak_gpu_mb", "peak_host_mb"] + [f"W{k}" for k in wk]
    agg = {}
    for key in keys:
        vals = [m[key] for m in per_seed.values() if m.get(key) is not None]
        if not vals:
            continue
        agg[key] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
            "values": vals,
        }
    return agg


def _fmt(agg, key, pct=False, prec=1):
    a = agg.get(key)
    if a is None:
        return "  -  "
    m, s = a["mean"], a["std"]
    if pct:
        m, s = 100 * m, 100 * s
    return f"{m:.{prec}f}±{s:.{prec}f}"


def write_markdown(summary, path, wk):
    rows = sorted(summary["configs"].items(),
                  key=lambda kv: (kv[1]["agg"].get("final_acc", {}).get("mean", -1)),
                  reverse=True)
    wk_cols = " | ".join(f"W{k}" for k in wk)
    lines = [
        f"# Final metrics — {summary['dataset']} (mean±std over seeds; acc/box shown as %)",
        "",
        f"Generated from `{summary['results_root']}`. full_matrix = all cells have the full N×N matrix.",
        "",
        f"| config | seeds | full | Final | Diag | Forget | FWT | {wk_cols} | Box(final) | train s |",
        "|" + "---|" * (9 + len(wk)),
    ]
    for tag, c in rows:
        a = c["agg"]
        nseed = c.get("n_seeds", 0)
        full = "yes" if all(m.get("full_matrix") for m in c["per_seed"].values()) and nseed else "no"
        wk_vals = " | ".join(_fmt(a, f"W{k}", pct=True) for k in wk)
        tt = a.get("train_time_sec", {})
        tt_s = f"{tt['mean']:.0f}" if tt else "-"
        lines.append(
            f"| {tag} | {nseed} | {full} | {_fmt(a,'final_acc',pct=True)} | "
            f"{_fmt(a,'diag',pct=True)} | {_fmt(a,'forget_bwt',pct=True)} | "
            f"{_fmt(a,'fwt',pct=True)} | {wk_vals} | {_fmt(a,'box_emission_final',pct=True)} | {tt_s} |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="symbol_qa")
    ap.add_argument("--results_root", default=None,
                    help="default: results/final/<dataset>")
    ap.add_argument("--ck_root", default=None,
                    help="default: checkpoints/final/<dataset> (for mem_log.jsonl)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--wk", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--out", default=None,
                    help="default: <results_root>/metrics_summary.json")
    ap.add_argument("--no_cache", action="store_true",
                    help="ignore + overwrite the per-cell _metrics.json caches")
    args = ap.parse_args()

    results_root = args.results_root or os.path.join("results", "final", args.dataset)
    ck_root = args.ck_root or os.path.join("checkpoints", "final", args.dataset)
    out_path = args.out or os.path.join(results_root, "metrics_summary.json")

    # discover configs = dirs holding seed_*/results.json
    tags = sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                   for p in glob.glob(os.path.join(results_root, "*", "seed_*", "results.json"))})
    if not tags:
        print(f"no results found under {results_root}/*/seed_*/results.json")
        return

    configs = {}
    for tag in tags:
        per_seed = {}
        for s in args.seeds:
            rp = os.path.join(results_root, tag, f"seed_{s}", "results.json")
            if not os.path.exists(rp):
                continue
            ck_seed = os.path.join(ck_root, tag, f"seed_{s}")
            print(f"[{tag} seed{s}] reading {rp}")
            per_seed[str(s)] = cell_metrics_cached(
                rp, ck_seed, args.wk, args.horizons,
                use_cache=not args.no_cache)
        if per_seed:
            configs[tag] = {"per_seed": per_seed, "n_seeds": len(per_seed),
                            "agg": aggregate(per_seed, args.wk)}

    summary = {"dataset": args.dataset, "results_root": results_root,
               "seeds": args.seeds, "wk": args.wk, "n_configs": len(configs),
               "configs": configs}
    os.makedirs(results_root, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    md_path = os.path.splitext(out_path)[0] + ".md"
    write_markdown(summary, md_path, args.wk)

    print(f"\n{len(configs)} configs -> {out_path}\n            -> {md_path}")
    # echo the human table
    with open(md_path) as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
