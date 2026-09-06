"""Randomly permute the TASK ORDER of a CL dataset (content unchanged).

Destroys innate task ordering (e.g. llm_qa's thematic runs — places, then guilds,
then treaties, ...) so the canonical on-disk order 0..N-1 carries no curriculum
structure. This is a ONE-TIME, fixed reshuffle of which topic-content sits at which
task index; it is orthogonal to train.py's --task_order_seed (which permutes at
read time on top of whatever the on-disk order is).

perm[i] = the ORIGINAL task index whose content becomes NEW task i. Content
(query / answer / topic / category / entity / train_text) moves intact; only the
POSITION markers (task_id, id) are rewritten to the new index. Fully reversible:
seed + perm + inverse are recorded in the output manifest. The global
query->answer map is re-verified single-valued afterward.

Usage:
    python data_generation/synthetic_qa/shuffle_task_order.py \
        --input_dir data/synthetic_qa/llm_qa \
        --output_dir data/synthetic_qa/llm_qa_shuffled --seed 0
"""
from __future__ import annotations
import argparse
import json
import os
import random
import shutil


def _load(p):
    with open(p) as f:
        return json.load(f)


def _dump(obj, p, indent=None):
    with open(p, "w") as f:
        if indent:
            json.dump(obj, f, indent=indent)
        else:
            json.dump(obj, f, separators=(", ", ": "))


def main():
    ap = argparse.ArgumentParser(description="Shuffle a dataset's task order (content unchanged).")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    man = _load(os.path.join(args.input_dir, "manifest.json"))
    n = man["n_tasks"]
    perm = random.Random(args.seed).sample(range(n), n)   # perm[i] = old idx at new pos i

    if os.path.exists(args.output_dir):
        raise SystemExit(f"refusing to overwrite existing {args.output_dir}")
    os.makedirs(args.output_dir)

    all_items = []
    for i in range(n):
        src = os.path.join(args.input_dir, f"task_{perm[i]}")
        dst = os.path.join(args.output_dir, f"task_{i}")
        shutil.copytree(src, dst)                          # copies ALL files verbatim
        for fn in ("train_items.json", "test_items.json"):
            fp = os.path.join(dst, fn)
            items = _load(fp)
            for idx, it in enumerate(items):
                if "task_id" in it:
                    it["task_id"] = i                     # position marker -> new index
                if "id" in it:
                    it["id"] = f"task_{i}_item_{idx}"      # position marker -> new index
                # topic / category / query / answer / entity / train_text = content, untouched
            _dump(items, fp)
        all_items.append(_load(os.path.join(dst, "test_items.json")))

    # ── manifest: reorder tasks[], update task_id, record the (reversible) shuffle ──
    man["tasks"] = [{**man["tasks"][perm[i]], "task_id": i} for i in range(n)]
    inv = [0] * n
    for new_pos, old in enumerate(perm):
        inv[old] = new_pos
    man["task_order_shuffle"] = {
        "seed": args.seed,
        "perm_old_at_new": perm,   # NEW task i holds ORIGINAL task perm[i]
        "inv_new_at_old": inv,     # ORIGINAL task j is now at NEW position inv[j]
        "source": os.path.basename(args.input_dir.rstrip("/")),
        "note": ("Task order randomized to remove innate ordering; content unchanged. "
                 "task_id/id rewritten to new index; topic/category identify content. "
                 "Revert by applying inv_new_at_old."),
    }
    _dump(man, os.path.join(args.output_dir, "manifest.json"), indent=2)

    # ── verify: content integrity (disambiguation must survive a pure permutation) ──
    q2a, nit = {}, 0
    for items in all_items:
        for it in items:
            q, a = it["query"], it["answer"]
            if q in q2a and q2a[q] != a:
                raise AssertionError(f"AMBIGUOUS {q!r}: {q2a[q]!r} vs {a!r}")
            q2a[q] = a
            nit += 1
    print(f"[ok] {n} tasks shuffled (seed {args.seed}); {nit} items, "
          f"{len(q2a)} unique queries, 0 ambiguous")
    print(f"  NEW task->topic (first 16): {[t['topic'] for t in man['tasks'][:16]]}")
    print(f"  perm[:16] (orig index now at pos 0..15): {perm[:16]}")


if __name__ == "__main__":
    main()
