"""Faithful re-implementation of O-LoRA (Wang et al., EMNLP-Findings 2023).

References:
  - Paper: https://aclanthology.org/2023.findings-emnlp.715/
           https://arxiv.org/abs/2310.14152
  - Official code: https://github.com/cmnfriend/O-LoRA  (forks PEFT)

The official implementation forks PEFT and modifies the `Linear` LoRA layer
to hold *two* matrix pairs per target module:

  - `lora_A`, `lora_B`: **frozen, accumulated** from all prior tasks. Their
    rank grows by `r` after each task (concatenation along rank axis at
    save time, see https://github.com/cmnfriend/O-LoRA/blob/main/src/peft/utils/save_and_load.py#L36-L52).
  - `loranew_A`, `loranew_B`: **trainable**, fixed at rank `r`. The new
    task's LoRA. Initialised Kaiming(A)/zeros(B) at start of each task
    (https://github.com/cmnfriend/O-LoRA/blob/main/src/peft/tuners/lora.py#L504-L506).

Forward (https://github.com/cmnfriend/O-LoRA/blob/main/src/peft/tuners/lora.py#L580-L601):
  out = base(x)
      + scale · lora_B(lora_A(x))         # frozen accumulated prior
      + scale · loranew_B(loranew_A(x))   # trainable current task

Orthogonality loss (https://github.com/cmnfriend/O-LoRA/blob/main/src/uie_trainer_lora.py#L91-L96):
  L_orth = Σ_layers Σ_ij |O_ij|,   O = lora_A · loranew_A^T  ∈ R^{r_sum × r}

  Important: the official code uses **entry-wise L1** norm of O, even
  though the paper (Eq. 7-8) writes it as `Σ |O[j,k]|²` (Frobenius
  squared). The code path is what they actually trained with — we match
  the code (L1) to be faithful to the published results.

L2 weight-decay term (https://github.com/cmnfriend/O-LoRA/blob/main/src/uie_trainer_lora.py#L99-L103):
  L_l2 = Σ_layers (||loranew_A||_2 + ||loranew_B||_2)
  (Default λ₂=0 in most configs, so usually a no-op. Some llama scripts
  use 0.1; we add the option but default to 0 like their main configs.)

Grad-accum quirk (https://github.com/cmnfriend/O-LoRA/blob/main/src/uie_trainer_lora.py#L83-L108):
  Only the CE loss is divided by `grad_accum`. The ortho and L2 terms
  are added at full magnitude after the division. We match this quirk
  exactly (it makes the effective ortho weight `λ₁ · grad_accum` larger
  than the textual λ₁ relative to CE — but that's the published recipe).
"""

from __future__ import annotations
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class OLoRALinear(nn.Module):
    """Drop-in replacement for `nn.Linear` implementing O-LoRA's two-pair
    LoRA structure. Forward sums frozen-prior + trainable-current
    contributions on top of the base linear's output.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        alpha: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        # Hold the original Linear; freeze its parameters
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        # Frozen accumulated prior LoRA — buffers (not in parameters()).
        # Empty at init; grow by `r` after every task.
        dt = base_linear.weight.dtype
        dev = base_linear.weight.device
        self.register_buffer(
            "lora_A", torch.empty(0, self.in_features, dtype=dt, device=dev))
        self.register_buffer(
            "lora_B", torch.empty(self.out_features, 0, dtype=dt, device=dev))

        # Trainable current-task LoRA — fixed rank r. Kaiming(A), zeros(B).
        self.loranew_A = nn.Parameter(
            torch.empty(r, self.in_features, dtype=dt, device=dev))
        self.loranew_B = nn.Parameter(
            torch.zeros(self.out_features, r, dtype=dt, device=dev))
        nn.init.kaiming_uniform_(self.loranew_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.base.weight, self.base.bias)
        x_drop = self.dropout(x)
        # Frozen prior contribution (if non-empty)
        if self.lora_A.shape[0] > 0:
            # (..., d_in) @ (d_in, r_sum) @ (r_sum, d_out) — via two linears
            out = out + self.scale * F.linear(
                F.linear(x_drop, self.lora_A), self.lora_B)
        # Trainable current contribution
        out = out + self.scale * F.linear(
            F.linear(x_drop, self.loranew_A), self.loranew_B)
        return out

    def fold_current_into_prior(self):
        """End-of-task fold: concat `loranew` into `lora` along the rank
        axis (rows of A, columns of B), then reset `loranew` to Kaiming/zero.
        Matches the `save_loranew=False` path in their save_and_load.py.
        """
        with torch.no_grad():
            new_A = torch.cat([self.lora_A, self.loranew_A.data], dim=0)
            new_B = torch.cat([self.lora_B, self.loranew_B.data], dim=1)
            # Replace buffers (need to use register_buffer to update shape)
            self.lora_A = new_A.to(self.lora_A.dtype)
            self.lora_B = new_B.to(self.lora_B.dtype)
            # Reset loranew for next task
            nn.init.kaiming_uniform_(self.loranew_A, a=math.sqrt(5))
            nn.init.zeros_(self.loranew_B)


def wrap_linears_with_olora(
    model,
    target_module_substrings: List[str],
    r: int,
    alpha: int,
    dropout: float = 0.0,
) -> List[str]:
    """Replace `nn.Linear` modules whose name ends with one of
    `target_module_substrings` (e.g. ['q_proj', 'k_proj']) with
    `OLoRALinear` wrappers. Returns the list of wrapped module names.

    "Replace" is module-TREE surgery only: the pretrained Linear is kept
    inside each wrapper (frozen, as `.base`) and used on every forward —
    no pretrained weights are removed or altered. With loranew_B = 0 at
    init, ΔW = 0 and the wrapped model is function-identical to the
    pretrained model (same as PEFT's LoraLayer wrapping).
    """
    to_replace = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if any(name.endswith(s) for s in target_module_substrings):
                to_replace.append((name, m))

    wrapped_names = []
    for name, base_lin in to_replace:
        parent_path, _, child_attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        wrapped = OLoRALinear(base_lin, r, alpha, dropout)
        setattr(parent, child_attr, wrapped)
        wrapped_names.append(name)
    return wrapped_names


def olora_orthogonality_loss(model) -> torch.Tensor:
    """L1 entry-wise norm of `lora_A · loranew_A^T`, summed over OLoRALinear
    layers. Faithful to https://github.com/cmnfriend/O-LoRA/blob/main/src/uie_trainer_lora.py#L91-L96.

    Note (paper vs code discrepancy): the paper text writes this as
    Frobenius-squared (Σ O[j,k]²); the official code uses entry-wise L1
    (Σ |O[j,k]|). We follow the code.
    """
    total = None
    for module in model.modules():
        if not isinstance(module, OLoRALinear):
            continue
        if module.lora_A.shape[0] == 0:
            continue  # No priors yet
        O = module.lora_A @ module.loranew_A.t()  # (r_sum, r)
        term = O.abs().sum()
        total = term if total is None else total + term
    if total is None:
        any_p = next(p for p in model.parameters() if p.requires_grad)
        return torch.zeros((), device=any_p.device, dtype=any_p.dtype)
    return total


def olora_l2_loss(model) -> torch.Tensor:
    """L2 norm (NOT squared) of `loranew_*` parameters summed across
    OLoRALinear layers. Faithful to
    https://github.com/cmnfriend/O-LoRA/blob/main/src/uie_trainer_lora.py#L99-L103.
    """
    total = None
    for module in model.modules():
        if not isinstance(module, OLoRALinear):
            continue
        for p in (module.loranew_A, module.loranew_B):
            term = torch.norm(p, p=2)
            total = term if total is None else total + term
    if total is None:
        any_p = next(p for p in model.parameters() if p.requires_grad)
        return torch.zeros((), device=any_p.device, dtype=any_p.dtype)
    return total


def fold_all_current_into_prior(model) -> int:
    """Call `fold_current_into_prior` on every OLoRALinear in the model.
    Returns count of layers folded.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, OLoRALinear):
            module.fold_current_into_prior()
            n += 1
    return n


# ── Disk-light per-task save + merge-at-eval (vs the full-model save) ────────
# O-LoRA's cumulative ΔW = scale · Σ_s (loranew_B_s · loranew_A_s) is low-rank,
# so storing the per-task rank-r `loranew` factors (a few hundred MB total) and
# summing them at eval is ~100× smaller than baking a full model per task.

def save_olora_loranew(model, ckpt_dir: str) -> int:
    """Save the CURRENT task's trainable `loranew_A/B` (rank r) per OLoRALinear,
    keyed by module name. Call BEFORE `fold_all_current_into_prior` (which
    resets loranew). The model after task t is reconstructed at eval by
    `olora_eval_merge` summing tasks 0..t. Returns #layers saved."""
    import json
    import os
    from safetensors.torch import save_file
    tensors = {}
    layers = []
    r = alpha = None
    for name, module in model.named_modules():
        if isinstance(module, OLoRALinear):
            tensors[name + ".A"] = module.loranew_A.detach().cpu().clone()
            tensors[name + ".B"] = module.loranew_B.detach().cpu().clone()
            layers.append(name)
            r, alpha = module.r, module.alpha
    os.makedirs(ckpt_dir, exist_ok=True)
    save_file(tensors, os.path.join(ckpt_dir, "olora_loranew.safetensors"))
    json.dump({"layers": layers, "r": r, "alpha": alpha, "scale": alpha / r},
              open(os.path.join(ckpt_dir, "olora_meta.json"), "w"))
    return len(layers)


def olora_eval_merge(base_model, checkpoint_dir: str, upto_t: int,
                     suffix: str = "", from_s: int = 0) -> int:
    """Fold O-LoRA's ΔW for tasks [`from_s` .. `upto_t`] into `base_model`'s
    linear weights, IN PLACE, on the model's OWN device:

        ΔW[layer] = scale · Σ_s (loranew_B_s · loranew_A_s)

    Two modes (same code path):

      * **one-shot** (`from_s=0`, default) — reconstruct the after-task-`upto_t`
        model on a fresh base. Used by `eval_math.py` / `eval_mcqa_loglik.py`
        and any single-checkpoint eval. (Back-compatible signature.)
      * **incremental** (`from_s = last_folded + 1`) — ADVANCE an
        already-merged model by only the *new* tasks. A checkpoint sweep that
        walks t = 0,1,…,N-1 in order folds each task's ΔW exactly ONCE and
        loads each factor file exactly once → **O(N)** total reconstruction
        work instead of the **O(N²)** of re-summing 0..t at every checkpoint.

    Both the rank-r matmul and the weight update run on `base_model.weight`'s
    device (GPU), avoiding a CPU reconstruction bottleneck.
    The B·A product is computed in fp32 (skinny, r≈16 — negligible) then cast
    to the weight dtype. Returns #layers folded for the last task's factor set.
    """
    import json
    import os
    from safetensors.torch import load_file
    n_layers = 0
    for s in range(from_s, upto_t + 1):
        d = os.path.join(checkpoint_dir, f"after_task_{s}{suffix}")
        fac = load_file(os.path.join(d, "olora_loranew.safetensors"))
        meta = json.load(open(os.path.join(d, "olora_meta.json")))
        scale = meta["scale"]
        n_layers = 0
        for name in meta["layers"]:
            lin = base_model.get_submodule(name)
            w = lin.weight
            A = fac[name + ".A"].to(device=w.device, dtype=torch.float32)  # [r, in]
            B = fac[name + ".B"].to(device=w.device, dtype=torch.float32)  # [out, r]
            dW = B @ A                                                     # [out, in]
            with torch.no_grad():
                w.data.add_((scale * dW).to(w.dtype))
            n_layers += 1
    return n_layers
