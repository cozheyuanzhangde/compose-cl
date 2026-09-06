"""Shared output-path helpers.

Keeps the per-seed run layout a property of train.py / evaluate.py themselves
rather than of any launcher script, so even a bare

    python train.py --output_dir checkpoints/foo --seed 7
    python evaluate.py  --checkpoint_dir checkpoints/foo --output_dir results/foo --seed 7

lands its checkpoints and results under a seed_<seed>/ leaf. Multiple seeds
then aggregate with a single glob (``<base>/seed_*/``) regardless of how the
run was launched.
"""
from __future__ import annotations
import os
import random
import re

# A directory whose basename looks like seed_42 / seed_007 / seed_foo.
_SEED_DIR_RE = re.compile(r"^seed_\w+$")


def seed_dir(base: str, seed) -> str:
    """Return ``<base>/seed_<seed>`` — the per-seed output/checkpoint dir.

    Idempotent: if ``base`` already ends in a ``seed_<...>`` leaf it is
    returned unchanged, so re-pointing a tool at an already-seeded dir (or a
    launcher that still appends the leaf itself) never double-nests. An empty
    ``base`` is returned unchanged.
    """
    if not base:
        return base
    leaf = os.path.basename(os.path.normpath(base))
    if _SEED_DIR_RE.match(leaf):
        return base
    return os.path.join(base, f"seed_{seed}")


def task_order(seed, n_full: int):
    """Stream-position -> disk-task-index map for continual-learning task shuffling.

    Returns a list ``perm`` of length ``n_full`` where stream position ``i`` reads
    the on-disk ``task_{perm[i]}``. ``seed`` None (or negative) -> identity, i.e.
    the canonical on-disk order 0..n_full-1 (behaviour byte-for-byte unchanged).

    The permutation is built over the FULL manifest horizon ``n_full`` (never a
    rung's capped ``n_tasks``); callers then use ``perm[:n_tasks]``. Because
    ``n_full`` is constant across rungs, ``perm[:k]`` is a prefix of ``perm[:k']``
    for k<k'. That prefix-stability is REQUIRED for TSH prefix-resume (a rung-1
    run over the first 10 stream positions must be a prefix of a rung-2 run over
    the first 20) and lets HP tuning run on a held-out task ordering disjoint from
    the canonical report order (pass a fixed dev seed to tune, omit it to report).
    """
    if seed is None or int(seed) < 0:
        return list(range(n_full))
    return random.Random(int(seed)).sample(range(n_full), n_full)
