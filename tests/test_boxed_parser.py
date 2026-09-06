r"""Robustness tests for the \boxed{} answer parser (evals/boxed_parse.py),
especially the multiple-box case the evaluator must get right.

Run: python -m pytest tests/test_boxed_parser.py -q
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evals.boxed_parse import extract_boxed, has_box, extract_primed  # noqa: E402


def test_single_box():
    assert extract_boxed(r"\boxed{IHH}") == "IHH"
    assert has_box(r"\boxed{IHH}") is True

def test_multiple_boxes_takes_first():
    # the model emits the answer first, then rambles more boxes
    assert extract_boxed(r"\boxed{IHH} \boxed{ZZZ} \boxed{QQQ}") == "IHH"
    assert extract_boxed("\\boxed{MID} \\]\nAnswer: \\boxed{SHT} \\]") == "MID"

def test_ramble_after_box():
    g = "\\boxed{IHH}\nQuestion: NEXT\nAnswer: \\boxed{XYZ}"
    assert extract_boxed(g) == "IHH"

def test_no_box():
    assert extract_boxed("IHH is the answer") is None
    assert has_box("no box here") is False

def test_unclosed_box_is_not_a_box():
    # documents current behavior: an unclosed \boxed{ does NOT count as emission
    assert extract_boxed(r"\boxed{IHH") is None
    assert has_box(r"\boxed{IHH") is False

def test_empty_box():
    assert extract_boxed(r"\boxed{}") == ""        # emitted format, empty content
    assert has_box(r"\boxed{}") is True

def test_whitespace_variants():
    assert extract_boxed(r"\boxed {IHH}") == "IHH"   # space before brace
    assert extract_boxed(r"\boxed{ IHH }") == "IHH"  # inner padding stripped

def test_leading_space_completion():
    # "Answer: \boxed{IHH}" -> completion after "Answer:" starts with a space
    assert extract_boxed(" \\boxed{IHH} trailing") == "IHH"

def test_garbage_multibox_first_still_returned():
    # real failure example: 4 single-char boxes, gold was "RRI" -> first="R"
    assert extract_boxed(r"\boxed{R} \boxed{P} \boxed{C} \boxed{F}") == "R"

def test_nested_braces_balanced():
    # balanced scanner: nested braces extract in FULL (no first-'}' truncation)
    assert extract_boxed(r"\boxed{a{b}}") == "a{b}"
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    # real_qa chemistry-subscript answers (the case that motivated the fix)
    assert extract_boxed(r"\boxed{Mg(NO_{3})_{2} + K_{2}CO_{3}}") == \
        "Mg(NO_{3})_{2} + K_{2}CO_{3}"
    # first balanced box wins, even with a nested one followed by ramble
    assert extract_boxed(r"\boxed{H_{2}O} \boxed{ZZZ}") == "H_{2}O"
    # an unclosed nested box is still "no closed box"
    assert extract_boxed(r"\boxed{a{b}") is None

def test_extract_primed():
    assert extract_primed("IHH}\nmore text") == ("IHH", True)
    assert extract_primed(" IHH } x") == ("IHH", True)
    assert extract_primed("IHH no brace") == ("IHH no brace", False)
    assert extract_primed("}") == ("", True)        # immediately closed -> empty
    # balanced: inner braces do not close the (already-open) box early
    assert extract_primed("Mg(NO_{3})_{2}} trailing") == ("Mg(NO_{3})_{2}", True)
    assert extract_primed(r"\frac{1}{2}} x") == (r"\frac{1}{2}", True)
