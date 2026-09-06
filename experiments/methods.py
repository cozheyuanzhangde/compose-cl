"""Canonical datasets, hyperparameters, and method compositions from the paper.

This module is the single source of truth for the final 21-method evaluation.
The first 16 methods form the 2^4 factorial over SI, self-distillation, replay,
and merged LoRA. The remaining five are online EWC, bare O-LoRA, bare
sequential OSRM, and two allocation-rule replacements in the TSH-selected
anchor stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence


MODEL = "Qwen/Qwen3-4B-Base"
SEEDS = (41, 42, 43)
FACTORS = ("si", "sd", "replay", "merge")


@dataclass(frozen=True)
class DatasetSpec:
    path: Path
    replay_weight: float
    replay_generation_temperature: float
    winner_anchors: Sequence[str]


DATASETS: Dict[str, DatasetSpec] = {
    "symbol_qa": DatasetSpec(
        path=Path("data/synthetic_qa/symbol_qa"),
        replay_weight=0.75,
        replay_generation_temperature=1.5,
        winner_anchors=("si", "sd", "replay"),
    ),
    "llm_qa": DatasetSpec(
        path=Path("data/synthetic_qa/llm_qa"),
        replay_weight=0.5,
        replay_generation_temperature=1.5,
        winner_anchors=("si", "replay"),
    ),
    "real_qa": DatasetSpec(
        path=Path("data/real_qa/qwen3_4b_base"),
        replay_weight=0.75,
        replay_generation_temperature=1.5,
        winner_anchors=("si", "replay"),
    ),
}


def _factorial_name(active: Sequence[str]) -> str:
    parts = [factor for factor in FACTORS if factor in active]
    return "_".join(parts) if parts else "vanilla"


def _factor_flags(dataset: str) -> Dict[str, List[str]]:
    spec = DATASETS[dataset]
    return {
        "si": ["--SI", "--si_lambda", "1", "--si_xi", "0.1"],
        "sd": [
            "--distill",
            "--distill_alpha",
            "1",
            "--distill_temperature",
            "5",
        ],
        "replay": [
            "--replay",
            "--replay_token",
            "--replay_n",
            "300",
            "--replay_weight",
            str(spec.replay_weight),
            "--replay_temperature",
            "2",
            "--replay_gen_temperature",
            str(spec.replay_generation_temperature),
            "--replay_batch_size",
            "32",
            "--replay_max_tokens",
            "384",
        ],
        "merge": ["--merge_lora_per_task"],
    }


def paper_methods(dataset: str) -> Dict[str, List[str]]:
    """Return the ordered 21-method final suite for ``dataset``."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")

    factor_flags = _factor_flags(dataset)
    methods: Dict[str, List[str]] = {}
    for size in range(len(FACTORS) + 1):
        for active in combinations(FACTORS, size):
            tag = _factorial_name(active)
            methods[tag] = [flag for factor in FACTORS if factor in active
                            for flag in factor_flags[factor]]

    methods["online_ewc"] = [
        "--online_ewc",
        "--ewc_lambda",
        "1000",
        "--online_ewc_gamma",
        "1",
        "--ewc_n_samples",
        "1000",
        "--online_ewc_normalize_fisher",
    ]
    methods["olora"] = [
        "--olora", "--olora_lambda", "0.5", "--olora_l2_lambda", "0"
    ]
    methods["osrm"] = ["--merge_lora_per_task", "--merge_lora_osrm"]

    anchors = DATASETS[dataset].winner_anchors
    anchor_flags = [flag for factor in anchors for flag in factor_flags[factor]]
    prefix = "_".join(anchors)
    methods[f"{prefix}_olora"] = anchor_flags + [
        "--olora",
        "--olora_lambda",
        "0.5",
        "--olora_l2_lambda",
        "0",
    ]
    methods[f"{prefix}_osrm"] = anchor_flags + [
        "--merge_lora_per_task",
        "--merge_lora_osrm",
    ]

    if len(methods) != 21:
        raise AssertionError(f"paper suite must contain 21 methods, got {len(methods)}")
    return methods


def is_merged(flags: Sequence[str]) -> bool:
    return "--merge_lora_per_task" in flags


def uses_replay_token(flags: Sequence[str]) -> bool:
    return "--replay_token" in flags
