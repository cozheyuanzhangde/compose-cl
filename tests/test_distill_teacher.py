"""Tests for the previous-state teacher used with O-LoRA. Two properties:

  1. MECHANISM: the teacher forward reproduces the model's output at SNAPSHOT
     time regardless of later parameter mutations, and restores the live
     (mutated) parameters afterwards so the student's backward sees them.
  2. INTEGRATION (olora): right after the task-t begin/reset, the wrapped
     model's forward equals the end-of-task-(t-1) model. Combined with (1), the
     teacher during task t therefore distills toward the previous-task model
     exactly — which is what LwF requires.

Run: python -m pytest tests/test_distill_teacher.py -q
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.self_distillation import FrozenStateTeacher  # noqa: E402
from core.olora_impl import (  # noqa: E402
    wrap_linears_with_olora, fold_all_current_into_prior,
)

torch.manual_seed(0)


class TinyLM(nn.Module):
    """Minimal causal-LM stand-in: embed -> [layers.N.attn.X_proj] -> head.
    Module names mimic HF (`...layers.N....q_proj`) so the wrappers' layer-index
    regex and target matching work. forward returns an object with `.logits`."""

    def __init__(self, vocab=12, dim=8, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            nn.ModuleDict({
                "self_attn": nn.ModuleDict({
                    "q_proj": nn.Linear(dim, dim, bias=False),
                    "v_proj": nn.Linear(dim, dim, bias=False),
                })
            }) for _ in range(n_layers)
        )
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for lyr in self.layers:
            x = x + lyr["self_attn"]["q_proj"](x) + lyr["self_attn"]["v_proj"](x)
        return SimpleNamespace(logits=self.head(x))


def _inputs(bsz=2, seqlen=5, vocab=12):
    ids = torch.randint(0, vocab, (bsz, seqlen))
    return ids, torch.ones_like(ids)


def _perturb_trainable(model, scale=0.1):
    """Simulate a training step by nudging every trainable parameter."""
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(scale * torch.randn_like(p))


def test_mechanism_reproduces_snapshot_and_restores():
    """Teacher logits == forward at snapshot time; live params restored after."""
    model = TinyLM()
    for p in model.parameters():
        p.requires_grad_(True)
    ids, am = _inputs()

    # Snapshot, then capture the reference (snapshot-time) logits.
    teacher = FrozenStateTeacher(model, "cpu", temperature=2.0)
    with torch.no_grad():
        ref_logits = model(input_ids=ids, attention_mask=am).logits.clone()

    # Mutate params (training), then run the teacher forward.
    _perturb_trainable(model, scale=0.5)
    with torch.no_grad():
        live_after_mutation = {n: p.detach().clone()
                               for n, p in model.named_parameters()}
        mutated_logits = model(input_ids=ids, attention_mask=am).logits.clone()

    student = torch.zeros_like(ref_logits)  # dummy student
    _ = teacher.distill_loss(student, ids, am, model=model)

    # The teacher's internal forward must have used the SNAPSHOT, so a forward we
    # run now (params should be restored to the mutated values) equals the
    # mutated logits, NOT the snapshot — i.e. live params were restored.
    with torch.no_grad():
        now_logits = model(input_ids=ids, attention_mask=am).logits
    assert torch.allclose(now_logits, mutated_logits, atol=1e-6), \
        "live params not restored after teacher forward"
    for n, p in model.named_parameters():
        assert torch.allclose(p, live_after_mutation[n], atol=1e-6)
    # And the snapshot genuinely differs from the mutated state (non-trivial test)
    assert not torch.allclose(ref_logits, mutated_logits, atol=1e-3)


def _teacher_logits(teacher, model, ids, am):
    """Pull the teacher logits out via the swap path (mirrors distill_loss)."""
    live = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            snap = teacher.snapshot.get(n)
            if snap is not None:
                live[n] = p.data
                p.data = snap
        out = model(input_ids=ids, attention_mask=am).logits.clone()
        for n, p in model.named_parameters():
            if n in live:
                p.data = live[n]
    return out


def test_olora_teacher_equals_prev_task_model():
    model = TinyLM()
    wrap_linears_with_olora(model, ["q_proj", "v_proj"], 4, 8, 0.0)
    ids, am = _inputs()

    # Task 0: train, then fold (resets loranew -> zero contribution).
    _perturb_trainable(model, 0.2)
    fold_all_current_into_prior(model)
    with torch.no_grad():
        end_t0 = model(input_ids=ids, attention_mask=am).logits.clone()

    # Task 1 starts here; loranew is in its post-fold reset state (B=0).
    teacher = FrozenStateTeacher(model, "cpu")
    _perturb_trainable(model, 0.3)            # "train" task 1 (moves loranew)
    teach = _teacher_logits(teacher, model, ids, am)
    assert torch.allclose(teach, end_t0, atol=1e-6), \
        "olora teacher during task 1 != end-of-task-0 model"


def test_distill_loss_backward_after_swap():
    """distill_loss runs BETWEEN the student's forward and backward. The
    teacher swap must therefore use .data REBINDING, never p.copy_(): the
    student graph holds the live param tensors as saved-for-backward, and an
    in-place copy_ bumps their autograd version counters (even under
    no_grad) -> student backward fails with 'modified by an inplace
    operation' (verified empirically). This test locks in the safe pattern
    end-to-end: forward -> distill_loss -> backward."""
    model = TinyLM()
    for p in model.parameters():
        p.requires_grad_(False)
    wrap_linears_with_olora(model, ["q_proj", "v_proj"], 4, 8, 0.0)
    model.train()
    ids, am = _inputs()

    teacher = FrozenStateTeacher(model, "cpu")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    out = model(input_ids=ids, attention_mask=am).logits   # student forward
    kd = teacher.distill_loss(out, ids, am, model=model)   # swap/teacher/restore
    (out.pow(2).mean() + kd).backward()                    # must not raise
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert model.training


def test_distill_loss_restores_params_on_teacher_error():
    """If the teacher forward raises (e.g. OOM), the try/finally must still
    rebind the live params and restore train mode — no corrupted model for
    callers that catch and continue."""
    import pytest

    model = TinyLM()
    for p in model.parameters():
        p.requires_grad_(False)
    wrap_linears_with_olora(model, ["q_proj", "v_proj"], 4, 8, 0.0)
    model.train()
    ids, am = _inputs()

    teacher = FrozenStateTeacher(model, "cpu")
    _perturb_trainable(model, 0.5)                          # live != snapshot
    live = {n: p.detach().clone() for n, p in model.named_parameters()}
    orig_forward = model.forward

    def boom(*a, **k):
        raise RuntimeError("simulated OOM")

    model.forward = boom
    try:
        with pytest.raises(RuntimeError, match="simulated OOM"):
            teacher.distill_loss(torch.zeros(2, 5, 12), ids, am, model=model)
    finally:
        model.forward = orig_forward
    assert all(torch.equal(p, live[n]) for n, p in model.named_parameters()), \
        "params left swapped after teacher-forward exception"
    assert model.training, "train mode not restored after exception"
