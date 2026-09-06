from __future__ import annotations

from typing import Dict, Tuple

import torch


class SynapticIntelligence:
    """Synaptic Intelligence (Zenke et al., ICML 2017).

    SI estimates a per-parameter importance online during training by
    accumulating the path integral ``-grad * delta_param``. At task boundaries,
    this path contribution is normalized by the squared task displacement and
    added to a cumulative quadratic penalty.

    State is kept in fp32 for numerical stability even when the model trains in
    bf16/fp16. Only parameters with ``requires_grad=True`` at ``begin_task`` are
    tracked for the current task.
    """

    def __init__(self, xi: float = 0.1, clamp_negative: bool = True):
        self.xi = float(xi)
        self.clamp_negative = bool(clamp_negative)

        self.importance: Dict[str, torch.Tensor] = {}
        self.params: Dict[str, torch.Tensor] = {}
        self._task_start: Dict[str, torch.Tensor] = {}
        self._omega: Dict[str, torch.Tensor] = {}
        self._step_task_grads: Dict[str, torch.Tensor] = {}

    def begin_task(self, model) -> None:
        """Snapshot the trainable parameters and reset the task path integral."""
        self._task_start = {}
        self._omega = {}
        self._step_task_grads = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p32 = p.detach().float().clone()
            self._task_start[name] = p32
            self._omega[name] = torch.zeros_like(p32)

    def record_task_gradients(self, model, task_loss: torch.Tensor) -> None:
        """Accumulate gradients of the unregularized task objective.

        The optimizer may still step on the base objective plus EWC/LwF/SI
        terms. SI's path integral, however, should use the base objective's
        gradients before those regularizers, matching the reference
        implementation's ``unreg_grads``.
        """
        if not self._task_start or not task_loss.requires_grad:
            return

        named_params = [
            (name, p)
            for name, p in model.named_parameters()
            if name in self._omega and p.requires_grad
        ]
        if not named_params:
            return

        names = [name for name, _p in named_params]
        params = [p for _name, p in named_params]
        grads = torch.autograd.grad(
            task_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )

        for name, grad in zip(names, grads):
            if grad is None:
                continue
            g = grad.detach().float()
            if name in self._step_task_grads:
                prev = self._step_task_grads[name].to(g.device)
                prev.add_(g)
                self._step_task_grads[name] = prev
            else:
                self._step_task_grads[name] = g.clone()

    def capture_step(self, model) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Capture pre-update parameters and unregularized task gradients.

        Call after AMP unscale and before gradient clipping / optimizer.step().
        The returned tensors are consumed by ``update_omega`` after the
        optimizer step has changed the parameters.
        """
        step_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        if not self._task_start or not self._step_task_grads:
            return step_state

        for name, p in model.named_parameters():
            task_grad = self._step_task_grads.get(name)
            if name not in self._omega or task_grad is None:
                continue
            step_state[name] = (
                p.detach().float().clone(),
                task_grad.detach().float().clone(),
            )
        return step_state

    @torch.no_grad()
    def update_omega(
        self,
        model,
        step_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Accumulate ``-grad * delta_param`` for one optimizer step."""
        if not step_state:
            self._step_task_grads = {}
            return

        params = dict(model.named_parameters())
        for name, (before, grad) in step_state.items():
            p = params.get(name)
            if p is None or name not in self._omega:
                continue
            delta = p.detach().float() - before
            self._omega[name] = self._omega[name].to(delta.device)
            self._omega[name].add_(-(grad.to(delta.device) * delta))
        self._step_task_grads = {}

    @torch.no_grad()
    def scale_pending_grads(self, k: float) -> None:
        """Rescale the current window's recorded task gradients in place.

        Used by the trainer's epoch-end flush: a partial accumulation window
        under-accumulates the (1/grad_accum)-scaled task grads, and the
        optimizer grads are rescaled by grad_accum/rem to a true window
        average — the SI path integral must see the same rescale or this
        step's ``-g·Δθ`` contribution to omega is under-counted.
        """
        for g in self._step_task_grads.values():
            g.mul_(k)

    @torch.no_grad()
    def consolidate(self, model) -> None:
        """Convert the current task path integral into cumulative importance."""
        if not self._task_start:
            return

        params = dict(model.named_parameters())
        for name, start in self._task_start.items():
            p = params.get(name)
            omega = self._omega.get(name)
            if p is None or omega is None:
                continue

            end = p.detach().float()
            start = start.to(end.device)
            omega = omega.to(end.device)
            delta = end - start
            task_importance = omega / (delta.pow(2) + self.xi)
            if self.clamp_negative:
                task_importance = task_importance.clamp_min(0.0)

            if name in self.importance:
                prev = self.importance[name].to(end.device)
                self.importance[name] = (prev + task_importance).detach().clone()
            else:
                self.importance[name] = task_importance.detach().clone()
            self.params[name] = end.detach().clone()

        self._task_start = {}
        self._omega = {}
        self._step_task_grads = {}

    def penalty(self, model) -> torch.Tensor:
        """Return ``sum_i Omega_i * (theta_i - theta_i_ref)^2``."""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        if not self.importance:
            return loss

        for name, p in model.named_parameters():
            if name not in self.importance or name not in self.params:
                continue
            importance = self.importance[name].to(p.device)
            ref = self.params[name].to(p.device)
            loss = loss + (importance * (p.float() - ref).pow(2)).sum()
        return loss
