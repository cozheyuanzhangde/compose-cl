"""Fast CPU checks for the paper's continual-learning mechanisms."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from core.ewc import OnlineEWC
from core.merge_lora_utils import osrm_orthogonal_init_A
from core.olora_impl import (
    OLoRALinear,
    fold_all_current_into_prior,
    olora_orthogonality_loss,
)
from core.si import SynapticIntelligence


class _Weight(nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(value.clone())


class _MockLoraLayer(nn.Module):
    """Small module with the PEFT attribute layout used by OSRM."""

    def __init__(self, a: torch.Tensor, out_features: int):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": _Weight(a)})
        self.lora_B = nn.ModuleDict(
            {"default": _Weight(torch.zeros(out_features, a.shape[0]))}
        )


class _Model(nn.Module):
    def __init__(self, **modules):
        super().__init__()
        for name, module in modules.items():
            setattr(self, name, module)


def test_olora_forward_and_fold():
    torch.manual_seed(1)
    base = nn.Linear(8, 6, bias=False)
    base_weight = base.weight.detach().clone()
    layer = OLoRALinear(base, r=2, alpha=4, dropout=0.0)
    model = _Model(proj=layer)
    inputs = torch.randn(3, 8)

    assert torch.allclose(layer(inputs), inputs @ base_weight.T, atol=1e-5)
    with torch.no_grad():
        layer.loranew_A.copy_(torch.randn_like(layer.loranew_A))
        layer.loranew_B.copy_(torch.randn_like(layer.loranew_B))
    a = layer.loranew_A.detach().clone()
    b = layer.loranew_B.detach().clone()
    expected = inputs @ base_weight.T + layer.scale * (inputs @ a.T @ b.T)
    assert torch.allclose(layer(inputs), expected, atol=1e-5)

    assert fold_all_current_into_prior(model) == 1
    assert layer.lora_A.shape == (2, 8)
    assert layer.lora_B.shape == (6, 2)
    assert torch.count_nonzero(layer.loranew_B) == 0


def test_olora_orthogonality_loss_detects_overlap():
    layer = OLoRALinear(nn.Linear(8, 6, bias=False), r=2, alpha=4)
    model = _Model(proj=layer)
    with torch.no_grad():
        layer.loranew_A.copy_(torch.randn_like(layer.loranew_A))
    fold_all_current_into_prior(model)

    prior = layer.lora_A.detach().double()
    q, _ = torch.linalg.qr(prior.T)
    candidate = torch.randn(2, 8, dtype=torch.float64)
    orthogonal = candidate - candidate @ q @ q.T
    with torch.no_grad():
        layer.loranew_A.copy_(orthogonal.to(layer.loranew_A.dtype))
    assert float(olora_orthogonality_loss(model).detach()) < 1e-3

    with torch.no_grad():
        layer.loranew_A.copy_(layer.lora_A)
    assert float(olora_orthogonality_loss(model).detach()) > 1e-3


def test_sequential_osrm_initializes_in_past_feature_null_space():
    torch.manual_seed(2)
    rank, input_size = 4, 32
    layer = _MockLoraLayer(torch.randn(rank, input_size), out_features=5)
    model = _Model(q_proj=layer)
    past = [{"q_proj": torch.randn(input_size)} for _ in range(2)]

    assert osrm_orthogonal_init_A(model, past) == 1
    new_a = layer.lora_A["default"].weight.detach().double()
    for features in past:
        assert (new_a @ features["q_proj"].double()).abs().max() < 1e-3


def test_online_ewc_penalty_is_quadratic_at_latest_anchor():
    model = _Model(weight=_Weight(torch.tensor([[2.0, 3.0]])))
    state = OnlineEWC(gamma=1.0, normalize_fisher=True)
    name = "weight.weight"
    state.fisher = {name: torch.tensor([[2.0, 0.5]])}
    state.params = {name: torch.tensor([[1.0, 1.0]])}
    expected = 2.0 * 1.0**2 + 0.5 * 2.0**2
    assert math.isclose(float(state.penalty(model).detach()), expected, rel_tol=1e-6)


def test_si_consolidates_positive_path_importance():
    model = _Model(weight=_Weight(torch.tensor([[1.0]])))
    state = SynapticIntelligence(xi=0.1, clamp_negative=True)
    state.begin_task(model)
    loss = model.weight.weight.square().sum()
    state.record_task_gradients(model, loss)
    step = state.capture_step(model)
    with torch.no_grad():
        model.weight.weight.sub_(0.2)
    state.update_omega(model, step)
    state.consolidate(model)

    name = "weight.weight"
    assert float(state.importance[name]) > 0
    assert float(state.penalty(model).detach()) == 0.0
    with torch.no_grad():
        model.weight.weight.add_(0.25)
    assert float(state.penalty(model).detach()) > 0
