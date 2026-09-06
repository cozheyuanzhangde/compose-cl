r"""Shared \boxed{} answer parser for the boxed-eval experiments.

Centralized and unit-tested so every evaluator uses identical parsing.

Contract:
  - extract_boxed: the content of the FIRST *closed* \boxed{...}, stripped, or None.
  - extract_primed: for prompts that already injected "\boxed{", the completion
    text up to the matching "}", with a flag for whether the brace was closed.

The scanner matches balanced nested braces, so brace-
containing answers — math (\boxed{\frac{1}{2}}) and real_qa chemistry
(\boxed{Mg(NO_{3})_{2}}) — extract in FULL. Brace-free answers (symbol_qa codes,
short factoids) are unaffected: the first '}' closes the (depth-1) box exactly as
before.
"""


def _find_balanced(text: str, open_idx: int):
    r"""Given text and the index of an opening '{', return the index of its
    matching '}' (balanced), or -1 if unclosed."""
    depth = 0
    for k in range(open_idx, len(text)):
        c = text[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return k
    return -1


def extract_boxed(text: str):
    r"""Return the first *closed* \boxed{...} content (balanced, stripped), or
    None. Multiple boxes -> the first (the trained answer position)."""
    n = len(text)
    i = text.find("\\boxed")
    while i != -1:
        j = i + 6                      # past "\boxed"
        while j < n and text[j].isspace():
            j += 1                     # allow "\boxed {...}"
        if j < n and text[j] == "{":
            close = _find_balanced(text, j)
            if close != -1:
                return text[j + 1:close].strip()
            # opener with no matching close -> not a closed box; keep scanning
        i = text.find("\\boxed", i + 6)
    return None


def has_box(text: str) -> bool:
    r"""Emission: did the text contain a parseable *closed* \boxed{...}?"""
    return extract_boxed(text) is not None


def extract_primed(completion: str):
    r"""For a prompt that already emitted the opener '\boxed{', parse the model's
    completion. Returns (content_stripped, closed) where closed=True iff the box
    was completed. Balanced: the opener counts as depth 1, so inner '{...}' do not
    close it early."""
    depth = 1
    for k, c in enumerate(completion):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return completion[:k].strip(), True
    return completion.strip(), False
