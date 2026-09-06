"""Integrity checks for the exact task streams shipped with the release."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STREAMS = {
    "symbol_qa": (ROOT / "data/synthetic_qa/symbol_qa", 100),
    "llm_qa": (ROOT / "data/synthetic_qa/llm_qa", 100),
    "real_qa": (ROOT / "data/real_qa/qwen3_4b_base", 50),
}


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_task_streams_are_complete_and_self_consistent():
    for root, items_per_task in STREAMS.values():
        manifest = _load(root / "manifest.json")
        assert manifest["n_tasks"] == 100
        assert {path.name for path in root.glob("task_*")} == {
            f"task_{index}" for index in range(100)
        }
        for task_index in range(100):
            task = root / f"task_{task_index}"
            train = _load(task / "train_items.json")
            test = _load(task / "test_items.json")
            texts = _load(task / "train_texts.json")
            assert len(train) == len(test) == len(texts) == items_per_task
            assert train == test
            rendered = [
                item.get(
                    "train_text",
                    f"Question: {item['query']}\nAnswer: {item['answer']}",
                )
                for item in train
            ]
            assert Counter(texts) == Counter(rendered)


def test_capability_benchmark_sizes_match_the_paper():
    expected = {
        "data/general_eval/math/gsm8k/eval.jsonl": 250,
        "data/general_eval/math/math/eval.jsonl": 243,
        "data/general_eval/mgsm/eval.jsonl": 1100,
        "data/general_eval/mmlu_redux/eval.jsonl": 3000,
    }
    for relative_path, expected_rows in expected.items():
        with (ROOT / relative_path).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        assert len(rows) == expected_rows
