"""Prompt construction shared by MMLU-Redux log-likelihood evaluation."""

from __future__ import annotations

import json
from pathlib import Path


LETTERS = ["A", "B", "C", "D", "E", "F"]

FEWSHOT = [
    (
        "What is the capital of France?",
        ["London", "Paris", "Rome", "Berlin"],
        1,
    ),
    (
        "Which gas do plants primarily absorb for photosynthesis?",
        ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"],
        2,
    ),
]


def format_question(question: str, choices) -> str:
    lines = [f"Question: {question}"]
    lines.extend(f"{LETTERS[index]}) {choice}" for index, choice in enumerate(choices))
    lines.append("Answer:")
    return "\n".join(lines)


def build_prefix() -> str:
    examples = [
        format_question(question, choices) + f" {LETTERS[answer]}"
        for question, choices, answer in FEWSHOT
    ]
    return "\n\n".join(examples) + "\n\n"


PREFIX = build_prefix()


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
