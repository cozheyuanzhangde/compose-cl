#!/usr/bin/env python3
"""Evaluate the paper's continual-learning checkpoints.

Each checkpoint is evaluated on the complete task stream. The lower triangle
of the resulting matrix gives final retention, immediate acquisition, and
forgetting; the first superdiagonal is retained for the appendix's forward-
transfer diagnostic. Evaluation rows are cached so interrupted long-horizon
runs can resume without repeating completed generation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.paths import seed_dir, task_order
from core.replay_token import REPLAY_TOKEN, add_replay_token
from evals.boxed_parse import extract_boxed, has_box


def _score_task(
    items: List[Dict],
    prompts: List[str],
    generations: List[str],
    generation_lengths: List[int],
    *,
    boxed_answers: bool,
) -> Tuple[float, List[Dict], Dict]:
    """Score one task and retain the generations needed to audit exact match."""
    records = []
    correct = 0
    boxes = 0
    for item, prompt, generation, length in zip(
        items, prompts, generations, generation_lengths
    ):
        if boxed_answers:
            parsed = extract_boxed(generation)
            hit = (
                parsed is not None
                and parsed.lower() == item["answer"].strip().lower()
            )
            boxes += int(has_box(generation))
        else:
            parsed = generation.split("\n", 1)[0].strip()
            hit = item["answer"].strip().lower() in generation.lower()
        correct += int(hit)
        records.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "prompt": prompt,
                "expected": item["answer"],
                "generated": generation,
                "parsed_answer": parsed,
                "correct": hit,
                "generation_length": length,
            }
        )

    n_items = len(items)
    accuracy = correct / n_items if n_items else 0.0
    stats = {
        "accuracy": accuracy,
        "n_items": n_items,
        "n_correct": correct,
        "box_emission": boxes / n_items if boxed_answers and n_items else None,
    }
    return accuracy, records, stats


@torch.inference_mode()
def evaluate_all_tasks(
    model,
    tokenizer,
    all_items: List[List[Dict]],
    device: str,
    *,
    max_new_tokens: int,
    batch_size: int,
    replay_prefix: str,
    boxed_answers: bool,
) -> List[Tuple[float, List[Dict], Dict]]:
    """Generate for all tasks in length-bucketed batches, then split by task."""
    model.eval()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    prompts = [
        replay_prefix + f"Question: {item['query']}\nAnswer:"
        for items in all_items
        for item in items
    ]
    if not prompts:
        return [
            _score_task([], [], [], [], boxed_answers=boxed_answers)
            for _ in all_items
        ]

    generations = [""] * len(prompts)
    generation_lengths = [0] * len(prompts)
    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in prompts
    ]
    order = sorted(range(len(prompts)), key=prompt_lengths.__getitem__)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in tqdm(
            range(0, len(order), batch_size),
            desc="    Generating (all tasks)",
            leave=False,
        ):
            indices = order[start : start + batch_size]
            batch_prompts = [prompts[i] for i in indices]
            encoded = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
            prompt_width = encoded["input_ids"].shape[1]
            for row, original_index in enumerate(indices):
                generated_ids = outputs[row][prompt_width:]
                generated_ids = generated_ids[generated_ids != pad_id]
                generations[original_index] = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
                generation_lengths[original_index] = len(generated_ids)
    finally:
        tokenizer.padding_side = original_padding_side

    task_rows = []
    cursor = 0
    for items in all_items:
        next_cursor = cursor + len(items)
        task_rows.append(
            _score_task(
                items,
                prompts[cursor:next_cursor],
                generations[cursor:next_cursor],
                generation_lengths[cursor:next_cursor],
                boxed_answers=boxed_answers,
            )
        )
        cursor = next_cursor
    return task_rows


def _eval_signature(args: argparse.Namespace, n_tasks: int) -> Dict:
    return {
        "model": args.model,
        "n_tasks": n_tasks,
        "max_new_tokens": args.max_new_tokens,
        "gen_batch_size": args.gen_batch_size,
        "boxed_answers": args.boxed_answers,
        "replay_token": args.replay_token,
        "task_order_seed": args.task_order_seed,
    }


def _row_cache_path(rows_dir: str, task: int) -> str:
    return os.path.join(rows_dir, f"after_task_{task}.json")


def _save_row_cache(rows_dir: str, task: int, payload: Dict) -> None:
    os.makedirs(rows_dir, exist_ok=True)
    path = _row_cache_path(rows_dir, task)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)


def _load_row_cache(rows_dir: str, task: int, signature: Dict):
    path = _row_cache_path(rows_dir, task)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    return payload if payload.get("signature") == signature else None


def _checkpoint_mtime(checkpoint_dir: str) -> float:
    latest = 0.0
    for root, _dirs, files in os.walk(checkpoint_dir):
        for filename in files:
            try:
                latest = max(latest, os.path.getmtime(os.path.join(root, filename)))
            except OSError:
                pass
    return round(latest, 3)


def _save_hyperparameters(
    output_dir: str, checkpoint_dir: str, eval_args: argparse.Namespace
) -> None:
    train_config = None
    train_config_path = os.path.join(checkpoint_dir, "train_config.json")
    if os.path.isfile(train_config_path):
        try:
            with open(train_config_path) as handle:
                train_config = json.load(handle)
        except (json.JSONDecodeError, OSError):
            pass
    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as handle:
        json.dump(
            {"train": train_config, "eval": vars(eval_args)},
            handle,
            indent=2,
            default=str,
        )


def _prune_intermediate_checkpoints(checkpoint_dir: str, n_tasks: int) -> None:
    """Keep only the final model after a complete matrix has been written."""
    removed = 0
    for task in range(n_tasks - 1):
        path = os.path.join(checkpoint_dir, f"after_task_{task}")
        if os.path.isdir(path):
            shutil.rmtree(path)
            removed += 1
    print(
        f"Pruned {removed} intermediate checkpoint(s); "
        f"kept after_task_{n_tasks - 1}/"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", default="data/synthetic_qa/symbol_qa")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gen_batch_size", type=int, default=128)
    parser.add_argument(
        "--boxed_answers",
        action="store_true",
        help="score the first generated \\boxed{...} by exact match",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replay_token",
        action="store_true",
        help="prepend the frozen replay token used by replay-trained checkpoints",
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=0,
        help="evaluate the first K stream tasks (0 means the full manifest)",
    )
    parser.add_argument(
        "--task_order_seed",
        type=int,
        default=None,
        help="use the same development-task permutation as training",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="ignore cached evaluation rows",
    )
    parser.add_argument(
        "--prune_checkpoints",
        action="store_true",
        help="after evaluation, keep only the final task checkpoint",
    )
    parser.add_argument(
        "--final_only",
        action="store_true",
        help="evaluate only the final checkpoint (used by TSH)",
    )
    parser.add_argument("--no_mem_log", action="store_true")
    return parser.parse_args()


def _mean(values: List[float]):
    return sum(values) / len(values) if values else None


def _paper_metrics(
    accuracy_matrix: List[List[float]],
    evaluated_checkpoints: List[int],
    n_tasks: int,
) -> Tuple[float, float | None, float | None, float | None]:
    """Return final retention, acquisition, forgetting, and forward transfer."""
    final_row = accuracy_matrix[-1]
    final_accuracy = _mean(final_row)
    if evaluated_checkpoints != list(range(n_tasks)):
        return final_accuracy, None, None, None
    immediate_accuracy = _mean(
        [accuracy_matrix[task][task] for task in range(n_tasks)]
    )
    forgetting = _mean(
        [
            max(accuracy_matrix[row][task] for row in range(task, n_tasks))
            - final_row[task]
            for task in range(n_tasks - 1)
        ]
    )
    forward_transfer = _mean(
        [accuracy_matrix[task - 1][task] for task in range(1, n_tasks)]
    )
    return final_accuracy, immediate_accuracy, forgetting, forward_transfer


def main() -> None:
    args = _parse_args()
    args.checkpoint_dir = seed_dir(args.checkpoint_dir, args.seed)
    args.output_dir = seed_dir(args.output_dir, args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.no_mem_log:
        from core.mem_log import init_mem_log

        init_mem_log(
            os.path.join(args.output_dir, "mem_log.jsonl"),
            meta={
                "script": "evaluate.py",
                "argv": sys.argv[1:],
                "model": args.model,
                "checkpoint_dir": args.checkpoint_dir,
            },
        )

    with open(os.path.join(args.data_dir, "manifest.json")) as handle:
        manifest = json.load(handle)
    n_tasks = manifest["n_tasks"]
    if args.n_tasks > 0:
        n_tasks = min(n_tasks, args.n_tasks)
    permutation = task_order(args.task_order_seed, manifest["n_tasks"])
    all_items = []
    for stream_task in range(n_tasks):
        path = os.path.join(
            args.data_dir,
            f"task_{permutation[stream_task]}",
            "test_items.json",
        )
        with open(path) as handle:
            all_items.append(json.load(handle))

    print(f"Loading base model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {
        "dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        "trust_remote_code": True,
        "device_map": device,
    }
    base_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    replay_prefix = ""
    replay_token_id = None
    if args.replay_token:
        replay_token_id = add_replay_token(tokenizer, base_model)
        replay_prefix = REPLAY_TOKEN

    accuracy_matrix: List[List[float]] = []
    detail_log: List[Dict] = []
    evaluated_checkpoints: List[int] = []
    rows_dir = os.path.join(args.output_dir, "_rows")
    base_signature = _eval_signature(args, n_tasks)

    olora_model = None
    olora_merged_through = -1

    for task in range(n_tasks):
        if args.final_only and task != n_tasks - 1:
            continue
        checkpoint_name = f"after_task_{task}"
        checkpoint = os.path.join(args.checkpoint_dir, checkpoint_name)
        if not os.path.isdir(checkpoint):
            print(f"Checkpoint {checkpoint} not found; skipping.")
            continue

        signature = {
            **base_signature,
            "checkpoint_mtime": _checkpoint_mtime(checkpoint),
        }
        cached = (
            None
            if args.overwrite
            else _load_row_cache(rows_dir, task, signature)
        )
        if cached is not None:
            print(f"[resume] {checkpoint_name}: using cached row")
            accuracy_matrix.append(cached["accuracy"])
            detail_log.extend(cached["detail"])
            evaluated_checkpoints.append(task)
            continue

        if not args.no_mem_log:
            from core.mem_log import get_mem_log

            get_mem_log().mark("eval_ckpt_start", checkpoint=checkpoint_name)

        is_adapter = os.path.isfile(os.path.join(checkpoint, "adapter_config.json"))
        is_olora = os.path.isfile(
            os.path.join(checkpoint, "olora_loranew.safetensors")
        )
        if is_adapter:
            model = PeftModel.from_pretrained(base_model, checkpoint)
        elif is_olora:
            from core.olora_impl import olora_eval_merge

            if olora_model is None:
                del base_model
                base_model = None
                torch.cuda.empty_cache()
                olora_model = AutoModelForCausalLM.from_pretrained(
                    args.model, **model_kwargs
                )
                if replay_token_id is not None:
                    embeddings = olora_model.get_input_embeddings().weight
                    if replay_token_id >= embeddings.shape[0]:
                        olora_model.resize_token_embeddings(len(tokenizer))
                        embeddings = olora_model.get_input_embeddings().weight
                    with torch.no_grad():
                        embeddings[replay_token_id] = embeddings[:replay_token_id].mean(
                            dim=0
                        )
                        output_embeddings = olora_model.get_output_embeddings()
                        if (
                            output_embeddings is not None
                            and output_embeddings.weight.data_ptr()
                            != embeddings.data_ptr()
                        ):
                            output_embeddings.weight[replay_token_id] = (
                                output_embeddings.weight[:replay_token_id].mean(dim=0)
                            )
            olora_eval_merge(
                olora_model,
                args.checkpoint_dir,
                task,
                from_s=olora_merged_through + 1,
            )
            olora_merged_through = task
            model = olora_model
        else:
            if base_model is not None:
                del base_model
                base_model = None
                torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)

        print(f"Evaluating {checkpoint_name} ...")
        task_results = evaluate_all_tasks(
            model,
            tokenizer,
            all_items,
            device,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch_size,
            replay_prefix=replay_prefix,
            boxed_answers=args.boxed_answers,
        )
        accuracy_row = []
        checkpoint_detail = []
        for eval_task, (accuracy, records, stats) in enumerate(task_results):
            tag = "current" if eval_task == task else (
                "past" if eval_task < task else "future"
            )
            print(
                f"  task {eval_task:03d} [{tag}]: {accuracy:.2%} "
                f"({stats['n_correct']}/{stats['n_items']})"
            )
            accuracy_row.append(accuracy)
            checkpoint_detail.append(
                {
                    "after_task": task,
                    "eval_task": eval_task,
                    "tag": tag,
                    "stats": stats,
                    "results": records,
                }
            )

        accuracy_matrix.append(accuracy_row)
        detail_log.extend(checkpoint_detail)
        evaluated_checkpoints.append(task)
        _save_row_cache(
            rows_dir,
            task,
            {
                "signature": signature,
                "after_task": task,
                "accuracy": accuracy_row,
                "detail": checkpoint_detail,
            },
        )

        if is_olora:
            model = None
        else:
            del model
            if is_adapter:
                del base_model
                torch.cuda.empty_cache()
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.model, **model_kwargs
                )
                if args.replay_token:
                    add_replay_token(tokenizer, base_model, verbose=False)
        gc.collect()
        torch.cuda.empty_cache()

    if not accuracy_matrix:
        raise SystemExit("No checkpoints were evaluated.")

    full_matrix = evaluated_checkpoints == list(range(n_tasks))
    final_accuracy, immediate_accuracy, forgetting, forward_transfer = (
        _paper_metrics(accuracy_matrix, evaluated_checkpoints, n_tasks)
    )

    output = {
        "accuracy_matrix": accuracy_matrix,
        "evaluated_checkpoints": evaluated_checkpoints,
        "avg_final_accuracy": final_accuracy,
        "avg_immediate_accuracy": immediate_accuracy,
        "avg_forgetting": forgetting,
        "avg_forward_transfer": forward_transfer,
        "detail_log": detail_log,
    }
    timing_path = os.path.join(args.checkpoint_dir, "train_timing.json")
    if os.path.isfile(timing_path):
        try:
            with open(timing_path) as handle:
                output.update(json.load(handle))
        except (json.JSONDecodeError, OSError):
            pass

    output_path = os.path.join(args.output_dir, "results.json")
    temporary = output_path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(output, handle, indent=2)
    os.replace(temporary, output_path)
    _save_hyperparameters(args.output_dir, args.checkpoint_dir, args)

    lines = ["ACCURACY MATRIX"]
    lines.append("After\\Eval  " + "  ".join(f"T{i}" for i in range(n_tasks)))
    for checkpoint_task, row in zip(evaluated_checkpoints, accuracy_matrix):
        lines.append(
            f"  T{checkpoint_task}        " + "  ".join(f"{value:.2f}" for value in row)
        )
    lines.extend(["", f"Final retention: {final_accuracy:.2%}"])
    if full_matrix:
        lines.append(f"Immediate acquisition: {immediate_accuracy:.2%}")
        lines.append(f"Forgetting: {forgetting:.2%}")
        lines.append(f"Forward transfer: {forward_transfer:.2%}")
    with open(os.path.join(args.output_dir, "results.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-4:]))
    print(f"Results saved to {output_path}")

    if args.prune_checkpoints:
        _prune_intermediate_checkpoints(args.checkpoint_dir, n_tasks)


if __name__ == "__main__":
    main()
