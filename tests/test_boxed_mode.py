"""--boxed_answers mode: train-side rendering <-> eval-side parsing contract.

The mode replaces pre-rendered boxed dataset variants: train.py wraps answers at load time via
core.datasets_local.box_answer_texts, and evaluate.py scores
by FIRST-\\boxed{} exact match via evals.boxed_parse.extract_boxed. This test
pins the round-trip: whatever the renderer writes, the parser must recover.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets_local import box_answer_texts          # noqa: E402
from evals.boxed_parse import extract_boxed, has_box      # noqa: E402


def test_render_basic():
    texts = ["Question: sym(QZXK)?\nAnswer: J7P",
             "Question: capital of France?\nAnswer: Paris"]
    out = box_answer_texts(texts)
    assert out[0] == "Question: sym(QZXK)?\nAnswer: \\boxed{J7P}"
    assert out[1] == "Question: capital of France?\nAnswer: \\boxed{Paris}"


def test_render_idempotent_and_passthrough():
    texts = ["Question: q?\nAnswer: ABC",
             "row without an answer marker",
             "Question: pre?\nAnswer: \\boxed{XYZ}"]
    once = box_answer_texts(texts)
    twice = box_answer_texts(once)
    assert twice == once, "double application must not double-wrap"
    assert once[1] == texts[1], "marker-less rows pass through unchanged"
    assert once[2] == texts[2], "pre-boxed rows pass through unchanged"


def test_render_first_marker_wins_and_seed_token_commutes():
    # seed-token prefix (--replay_token) precedes the marker; transform
    # must still find it (partition is on the internal '\nAnswer:').
    t = "<seed>Question: q?\nAnswer: K2M"
    assert box_answer_texts([t])[0] == "<seed>Question: q?\nAnswer: \\boxed{K2M}"


def test_round_trip_with_eval_parser():
    gold = ["J7P", "Paris", "42"]
    texts = [f"Question: q{i}?\nAnswer: {g}" for i, g in enumerate(gold)]
    for tx, g in zip(box_answer_texts(texts), gold):
        completion = tx.split("\nAnswer:", 1)[1]
        assert has_box(completion)
        assert extract_boxed(completion) == g, "renderer->parser round-trip"


def test_eval_scoring_rules():
    # The hit rule used by evaluate.py/_score_task:
    def hit(gen, answer):
        box = extract_boxed(gen)
        return box is not None and box.lower() == answer.lower().strip()

    assert hit(" \\boxed{J7P} and more rambling", "J7P")
    assert hit(" \\boxed{j7p}", "J7P"), "case-insensitive"
    assert not hit(" J7P (no box emitted)", "J7P"), "no box => miss, even if correct in plain text"
    assert not hit(" \\boxed{WRONG} then \\boxed{J7P}", "J7P"), "FIRST box wins (boxed_parse contract)"
    assert not hit(" \\boxed{", "J7P"), "unclosed box => miss"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} boxed-mode tests passed")
