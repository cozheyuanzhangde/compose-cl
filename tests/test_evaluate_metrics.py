"""Paper metric definitions computed from the temporal accuracy matrix."""

import math

from evaluate import _paper_metrics


def test_full_matrix_metrics():
    matrix = [
        [1.0, 0.1, 0.2],
        [0.6, 0.9, 0.3],
        [0.4, 0.7, 0.8],
    ]
    final, immediate, forgetting, forward = _paper_metrics(
        matrix, [0, 1, 2], 3
    )
    assert math.isclose(final, 1.9 / 3)
    assert math.isclose(immediate, 0.9)
    assert math.isclose(forgetting, 0.4)
    assert math.isclose(forward, 0.2)


def test_tsh_final_only_has_final_retention_only():
    final, immediate, forgetting, forward = _paper_metrics(
        [[0.2, 0.4, 0.6]], [2], 3
    )
    assert math.isclose(final, 0.4)
    assert immediate is None
    assert forgetting is None
    assert forward is None
