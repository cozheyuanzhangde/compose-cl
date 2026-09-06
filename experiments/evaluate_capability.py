#!/usr/bin/env python3
"""Evaluate a final checkpoint on the four held-out capability benchmarks."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.methods import DATASETS, MODEL, SEEDS, paper_methods  # noqa: E402
MATH_BENCHMARKS = {
    "gsm8k": Path("data/general_eval/math/gsm8k/eval.jsonl"),
    "math": Path("data/general_eval/math/math/eval.jsonl"),
    "mgsm": Path("data/general_eval/mgsm/eval.jsonl"),
}
MCQA_BENCHMARK = Path("data/general_eval/mmlu_redux/eval.jsonl")


def _run(command: List[str], output: Path, *, force: bool, dry_run: bool) -> None:
    if output.is_file() and not force:
        print(f"skip: {output} exists", flush=True)
        return
    print("$ " + shlex.join(command), flush=True)
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=REPO, check=True)


def _checkpoint_args(args: argparse.Namespace) -> tuple[List[str], List[str], Path]:
    """Return math args, MCQA args, and the output directory."""
    if args.method == "base_model":
        output = _rooted(args.result_root) / args.dataset / "_base_model" / "general_eval"
        return ["--base_model", args.model], ["--model", args.model], output

    methods = paper_methods(args.dataset)
    if args.method not in methods:
        raise SystemExit(
            f"unknown method {args.method!r}; choose base_model or one of: "
            + ", ".join(methods)
        )
    flags = methods[args.method]
    seed_root = _rooted(args.checkpoint_root) / args.dataset / args.method / f"seed_{args.seed}"
    final_checkpoint = seed_root / "after_task_99"
    if not final_checkpoint.is_dir() and not args.dry_run:
        raise SystemExit(f"missing final checkpoint: {final_checkpoint}")

    if "--olora" in flags:
        math_args = [
            "--base_model", args.model,
            "--olora_factors", str(seed_root),
            "--olora_upto", "99",
        ]
        mcqa_args = [
            "--model", args.model,
            "--olora_factors", str(seed_root),
            "--olora_upto", "99",
        ]
    elif "--merge_lora_per_task" not in flags:
        math_args = ["--base_model", args.model, "--adapter", str(final_checkpoint)]
        mcqa_args = ["--model", args.model, "--adapter", str(final_checkpoint)]
    else:
        math_args = ["--base_model", str(final_checkpoint)]
        mcqa_args = ["--model", str(final_checkpoint)]

    output = (
        _rooted(args.result_root)
        / args.dataset
        / args.method
        / "general_eval"
        / f"seed_{args.seed}"
    )
    return math_args, mcqa_args, output


def _rooted(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--method", required=True, help="paper method tag or base_model")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--checkpoint-root", default="checkpoints/final")
    parser.add_argument("--result-root", default="results/final")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=(*MATH_BENCHMARKS, "mmlu_redux"),
        default=[*MATH_BENCHMARKS, "mmlu_redux"],
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    math_args, mcqa_args, output_dir = _checkpoint_args(args)
    for name in args.benchmarks:
        output = output_dir / f"{name}.json"
        if name in MATH_BENCHMARKS:
            command = [
                args.python,
                "evals/eval_math.py",
                *math_args,
                "--eval_jsonl", str(MATH_BENCHMARKS[name]),
                "--grader", "math_verify",
                "--max_new_tokens", "512",
                "--batch_size", "64",
                "--output", str(output),
            ]
        else:
            command = [
                args.python,
                "evals/eval_mcqa_loglik.py",
                *mcqa_args,
                "--eval_jsonl", str(MCQA_BENCHMARK),
                "--batch_size", "32",
                "--output", str(output),
            ]
        _run(command, output, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
