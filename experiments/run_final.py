#!/usr/bin/env python3
"""Run one or more cells from the paper's canonical 21-method evaluation.

The launcher is intentionally scheduler-agnostic. Run one process per assigned
GPU (or let a cluster scheduler launch independent method/seed cells). Training
resumes at task boundaries, and evaluation resumes at matrix-row boundaries.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.methods import (  # noqa: E402
    DATASETS,
    MODEL,
    SEEDS,
    is_merged,
    paper_methods,
    uses_replay_token,
)


def _rooted(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def _run(command: List[str], *, dry_run: bool) -> None:
    print("$ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO, check=True)


def _selected_methods(dataset: str, requested: Iterable[str]) -> List[str]:
    available = paper_methods(dataset)
    requested = list(requested)
    if requested == ["all"]:
        return list(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise SystemExit(
            f"unknown method(s) for {dataset}: {unknown}\n"
            f"available: {', '.join(available)}"
        )
    return requested


def run_cell(args: argparse.Namespace, dataset: str, method: str, seed: int) -> None:
    spec = DATASETS[dataset]
    flags = paper_methods(dataset)[method]
    checkpoint_root = _rooted(args.checkpoint_root) / dataset / method
    result_root = _rooted(args.result_root) / dataset / method
    seed_checkpoint = checkpoint_root / f"seed_{seed}"
    seed_result = result_root / f"seed_{seed}"
    final_checkpoint = seed_checkpoint / "after_task_99"
    result_file = seed_result / "results.json"

    common = [
        "--model", args.model,
        "--data_dir", str(spec.path),
        "--n_tasks", "100",
        "--seed", str(seed),
        "--lr", "5e-4",
        "--epochs", "10",
        "--batch_size", "8",
        "--grad_accum", "1",
        "--max_seq_len", "384",
        "--lora_r", "32",
        "--lora_alpha", "64",
        "--lora_dropout", "0.05",
        "--lora_target_modules",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--weight_decay", "0.01",
        "--warmup_frac", "0.05",
        "--boxed_answers",
    ]

    should_train = not final_checkpoint.is_dir()
    # A completed checkpoint with a surviving resume sidecar means training was
    # interrupted during finalization; re-enter so train.py can finish cleanly.
    if (seed_checkpoint / "resume" / "state.pt").is_file() and not result_file.is_file():
        should_train = True

    if args.stage in ("all", "train"):
        if should_train or args.dry_run:
            _run([
                args.python,
                "train.py",
                *common,
                "--output_dir",
                str(checkpoint_root),
                *flags,
            ], dry_run=args.dry_run)
        else:
            print(f"skip train: {dataset}/{method}/seed_{seed} is complete", flush=True)

    if args.stage in ("all", "evaluate"):
        if not result_file.is_file() or args.overwrite_eval or args.dry_run:
            command = [
                args.python,
                "evaluate.py",
                "--model", args.model,
                "--data_dir", str(spec.path),
                "--checkpoint_dir", str(checkpoint_root),
                "--output_dir", str(result_root),
                "--n_tasks", "100",
                "--seed", str(seed),
                "--gen_batch_size", "256",
                "--max_new_tokens", "256",
                "--boxed_answers",
            ]
            if uses_replay_token(flags):
                command.append("--replay_token")
            if is_merged(flags) and not args.keep_intermediate_checkpoints:
                command.append("--prune_checkpoints")
            if args.overwrite_eval:
                command.append("--overwrite")
            _run(command, dry_run=args.dry_run)
        else:
            print(f"skip eval: {result_file} exists", flush=True)

    # Per-row JSON files only exist to resume an incomplete matrix. The final
    # results.json contains the same rows and is the analysis source of truth.
    row_cache = seed_result / "_rows"
    if (not args.dry_run and result_file.is_file() and row_cache.is_dir()
            and not args.keep_row_cache):
        shutil.rmtree(row_cache)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--method",
        dest="methods",
        nargs="+",
        default=["all"],
        help="one or more method tags, or 'all' (default)",
    )
    parser.add_argument("--seed", dest="seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--checkpoint-root", default="checkpoints/final")
    parser.add_argument("--result-root", default="results/final")
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--overwrite-eval", action="store_true")
    parser.add_argument("--keep-intermediate-checkpoints", action="store_true")
    parser.add_argument("--keep-row-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    methods = _selected_methods(args.dataset, args.methods)
    for seed in args.seeds:
        for method in methods:
            print(f"\n=== {args.dataset} / {method} / seed {seed} ===", flush=True)
            run_cell(args, args.dataset, method, seed)


if __name__ == "__main__":
    main()
