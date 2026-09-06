#!/usr/bin/env python
"""Build the canonical globally remixed Real-QA stream from a blocked stream.

The source stream `rebuilt_data/real_qa_blocked/<model>/` is 100 tasks in fixed
alphabetical order, each task = 50 rows from a SINGLE dataset (task_0..9
arc_challenge, task_10..19 arc_easy, ... task_90..99 webquestions). That gives a
strong second-order structure: task position determines dataset, adjacent tasks
are same-domain, and the format shifts every 10 tasks.

This script produces the canonical `data/real_qa/<model>/` where that structure
is destroyed: pool all 5,000 rows, shuffle globally (fixed seed), re-chunk into 100
tasks of 50. Each new task is a uniform random MIX of all 10 datasets (~5 rows
each) drawn from ONE pooled distribution -> tasks become exchangeable i.i.d.
samples, like data/synthetic_qa/. Same rows, same 100x50 shape, test==train;
ONLY the row->task assignment changes.

Determinism: rows are canonically ordered by their full JSON (content-determined,
independent of file read order) BEFORE the seeded shuffle, so `build` reproduces
byte-for-byte. Per-row fields (id, query, answer, answer_aliases, dataset, type)
are carried through untouched, so eval scoring (answer + aliases) still works and
each row still records which dataset it came from.

Usage:
    python data_generation/real_qa/remix_real_qa.py \
        --src_root rebuilt_data/real_qa_blocked \
        --out_root rebuilt_data/real_qa --seed 42
    # By default, process every model directory under src_root; --n is 50.
Never writes under src_root.
"""
from __future__ import annotations
import argparse, json, os, random
from collections import Counter

TEXT_TMPL = "Question: {query}\nAnswer: {answer}"


def _load_model_pool(src_model_dir: str) -> list[dict]:
    """All train_items across task_0..N-1, as a flat list (order irrelevant —
    re-canonicalized before shuffling)."""
    tasks = sorted(
        (d for d in os.listdir(src_model_dir) if d.startswith("task_")),
        key=lambda d: int(d.split("_")[1]))
    if not tasks:
        raise SystemExit(f"no task_* under {src_model_dir}")
    pool = []
    for tname in tasks:
        with open(os.path.join(src_model_dir, tname, "train_items.json")) as f:
            pool.extend(json.load(f))
    return pool


def remix_model(src_model_dir: str, out_model_dir: str, model_name: str,
                seed: int, n: int) -> dict:
    pool = _load_model_pool(src_model_dir)
    if len(pool) % n != 0:
        raise SystemExit(
            f"{model_name}: pool size {len(pool)} not divisible by n={n}; "
            f"refusing to silently drop rows")
    n_tasks = len(pool) // n

    # Canonical, content-determined order (no dependence on read/dict order),
    # then a single seeded global shuffle => bit-reproducible.
    pool.sort(key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=True))
    random.Random(seed).shuffle(pool)

    os.makedirs(out_model_dir, exist_ok=True)
    tasks_meta = []
    for t in range(n_tasks):
        items = pool[t * n:(t + 1) * n]
        texts = [TEXT_TMPL.format(query=it["query"], answer=it["answer"])
                 for it in items]
        tdir = os.path.join(out_model_dir, f"task_{t}")
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, "train_items.json"), "w") as f:
            json.dump(items, f, ensure_ascii=False)
        with open(os.path.join(tdir, "train_texts.json"), "w") as f:
            json.dump(texts, f, ensure_ascii=False)
        with open(os.path.join(tdir, "test_items.json"), "w") as f:  # test==train
            json.dump(items, f, ensure_ascii=False)
        comp = Counter(it["dataset"] for it in items)
        tasks_meta.append({"index": t, "n_train": n, "n_eval": n,
                           "dataset_composition": dict(sorted(comp.items()))})

    global_counts = Counter(it["dataset"] for it in pool)
    manifest = {
        "dataset_type": "cl_qa",
        "model_name": model_name,
        "variant": "full_remix_iid",
        "n_tasks": n_tasks, "uniform_n": n,
        "n_datasets": len(global_counts),
        "test_equals_train": True,
        "remix_seed": seed,
        "source": src_model_dir,
        "method": ("pooled all rows -> canonical sort by full-JSON -> global "
                   f"seeded shuffle (seed {seed}) -> re-chunk into {n_tasks} "
                   f"tasks of {n}; each task is a random i.i.d. mix of datasets "
                   "(no per-task domain; second-order task structure removed)"),
        "dataset_counts": dict(sorted(global_counts.items())),
        "tasks_in_order": tasks_meta,
    }
    with open(os.path.join(out_model_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_root", default="rebuilt_data/real_qa_blocked")
    ap.add_argument("--out_root", default="rebuilt_data/real_qa")
    ap.add_argument("--models", nargs="*", default=None,
                    help="default: every <model>/ under src_root that has task_0/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=50, help="rows per task")
    args = ap.parse_args()

    src_root = os.path.abspath(args.src_root)
    out_root = os.path.abspath(args.out_root)
    if out_root == src_root or out_root.startswith(src_root + os.sep):
        raise SystemExit("refusing to write inside src_root (would clobber the "
                         "committed dataset)")

    if args.models:
        models = args.models
    else:
        models = sorted(
            d for d in os.listdir(src_root)
            if os.path.isdir(os.path.join(src_root, d, "task_0")))
    if not models:
        raise SystemExit(f"no models with task_0/ under {src_root}")

    os.makedirs(out_root, exist_ok=True)
    print(f"remix: {src_root} -> {out_root}  seed={args.seed}  n={args.n}")
    print(f"models: {models}\n")
    for m in models:
        man = remix_model(os.path.join(src_root, m), os.path.join(out_root, m),
                          m, args.seed, args.n)
        comp0 = man["tasks_in_order"][0]["dataset_composition"]
        print(f"  {m}: {man['n_tasks']} tasks x {man['uniform_n']}  "
              f"counts={man['dataset_counts']}")
        print(f"      task_0 mix = {comp0}")
    print(f"\nDONE -> {out_root}")


if __name__ == "__main__":
    main()
