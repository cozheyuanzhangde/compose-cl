"""Paper-level invariants for the public experiment launchers."""

from __future__ import annotations

import ast
from pathlib import Path

from experiments.methods import DATASETS, FACTORS, SEEDS, paper_methods
from tsh.run import expand, load_cfg, select


ROOT = Path(__file__).resolve().parents[1]


def test_final_suite_has_exact_factorial_and_comparisons():
    expected_factorial = {
        "_".join(factor for index, factor in enumerate(FACTORS) if mask & (1 << index))
        or "vanilla"
        for mask in range(2 ** len(FACTORS))
    }

    for dataset in DATASETS:
        methods = paper_methods(dataset)
        assert len(methods) == 21
        assert expected_factorial.issubset(methods)
        assert {"online_ewc", "olora", "osrm"}.issubset(methods)
        assert len(set(methods)) == len(methods)


def test_registered_replay_hyperparameters_match_the_paper():
    expected_weights = {"symbol_qa": 0.75, "llm_qa": 0.5, "real_qa": 0.75}
    assert SEEDS == (41, 42, 43)
    for dataset, weight in expected_weights.items():
        spec = DATASETS[dataset]
        assert spec.replay_weight == weight
        assert spec.replay_generation_temperature == 1.5


def test_tsh_pool_and_funnel_are_exact():
    for dataset in DATASETS:
        cfg = load_cfg(ROOT / "tsh" / "configs" / f"{dataset}.yaml")
        candidates = expand(cfg)
        assert len(candidates) == 90
        assert len({candidate["tag"] for candidate in candidates}) == 90
        assert cfg["rungs"] == [10, 20, 50, 100]
        assert cfg["seeds"] == [41, 42, 43]
        assert "--task_order_seed 1234" in cfg["common_flags"]

        counts = [len(candidates)]
        survivors = candidates
        ranked = candidates  # only the length matters for deterministic cuts
        for rung_index, next_budget in enumerate(cfg["rungs"][1:]):
            survivors = select(cfg, ranked, rung_index, next_budget)
            counts.append(len(survivors))
            ranked = survivors
        assert counts == [90, 45, 23, 10]


def test_final_runs_use_the_canonical_task_order():
    for dataset in DATASETS:
        for flags in paper_methods(dataset).values():
            assert "--task_order_seed" not in flags


def _declared_options(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("--")
    }


def test_registered_flags_exist_in_the_training_cli():
    declared = _declared_options(ROOT / "train.py")
    common = {
        "--model", "--data_dir", "--output_dir", "--n_tasks", "--seed",
        "--lr", "--epochs", "--batch_size", "--grad_accum", "--max_seq_len",
        "--lora_r", "--lora_alpha", "--lora_dropout", "--lora_target_modules",
        "--weight_decay", "--warmup_frac", "--boxed_answers",
    }
    assert common.issubset(declared)
    for dataset in DATASETS:
        for flags in paper_methods(dataset).values():
            assert {flag for flag in flags if flag.startswith("--")}.issubset(declared)
