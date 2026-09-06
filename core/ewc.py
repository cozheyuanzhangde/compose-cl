from __future__ import annotations
from typing import Dict, List, Optional

import torch
from tqdm import tqdm


class _FisherEstimate:
    def __init__(self, model, tokenizer, texts: List[str], device: str,
                 max_seq_len: int = 256, n_samples: int = 200):
        self.params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
        self._compute(model, tokenizer, texts, device, max_seq_len, n_samples)

    def _compute(self, model, tokenizer, texts, device, max_seq_len, n_samples):
        model.eval()
        fisher_acc: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                # C3 fix: accumulate Fisher in fp32 (model is bf16; summing
                # ~1e-6-scale squared grads over n_samples in bf16 loses ~18%).
                fisher_acc[n] = torch.zeros_like(p, dtype=torch.float32)

        subset = texts[:n_samples] if len(texts) > n_samples else texts
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        for t in tqdm(subset, desc="    Fisher estimation", leave=False):
            model.zero_grad()
            enc = tokenizer(t, max_length=max_seq_len, truncation=True,
                            return_tensors="pt", add_special_tokens=True).to(device)
            ids = enc["input_ids"]
            out = model(input_ids=ids, attention_mask=enc["attention_mask"], labels=ids)
            out.loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_acc[n] += p.grad.detach().float().pow(2)

        for n in fisher_acc:
            self.fisher[n] = fisher_acc[n] / len(subset)

        for n, p in model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.detach().clone()

        model.zero_grad()

    def penalty(self, model) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss


class OnlineEWC:
    """Online EWC with one running Fisher and one latest parameter anchor.

    This follows the online EWC update from Schwarz et al.:
        F*_t = gamma * F*_{t-1} + F_t

    The penalty for the next task uses gamma * F*_{t-1}, matching the paper's
    graceful forgetting term. Persistent storage is constant in the number of
    tasks: one Fisher tensor and one parameter snapshot per trainable tensor.

    By default each task Fisher is rescaled to a common mean before the running
    accumulation (normalize_fisher=True), implementing the equal-task-weighting
    of Section 4 of Progress & Compress so that tasks with larger raw Fisher
    norm do not dominate the running term. The common mean is anchored to the
    first task's raw Fisher mean (not 1.0) so the running Fisher stays on the
    raw-Fisher scale and --ewc_lambda means the same thing for EWC and online
    EWC; normalizing to unit mean instead inflated the penalty by ~1/mean(F)
    (1e4-1e5x), which pinned every parameter at the first anchor and collapsed
    the accuracy-matrix diagonal.
    """

    def __init__(self, gamma: float = 1.0, normalize_fisher: bool = True):
        if gamma < 0.0 or gamma > 1.0:
            raise ValueError("OnlineEWC gamma must be in [0, 1]")
        self.gamma = float(gamma)
        self.normalize_fisher = bool(normalize_fisher)
        self.params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
        self.n_tasks = 0
        # Common mean every task Fisher is rescaled to, set from the first
        # task's raw Fisher mean. Anchors the running Fisher to the raw-Fisher
        # scale so --ewc_lambda is consistent with plain EWC.
        self._ref_mean: Optional[float] = None

    def __bool__(self) -> bool:
        return bool(self.fisher)

    def _normalize(self, fisher: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.normalize_fisher:
            return fisher

        total = 0.0
        n_elem = 0
        for f in fisher.values():
            f32 = f.detach().float()
            total += f32.sum().item()
            n_elem += f32.numel()

        if total <= 0.0 or n_elem == 0:
            return fisher

        # Rescale this task's Fisher to a common mean, anchored to the first
        # task's raw Fisher mean. Preserves relative parameter importance within
        # the task and gives every task comparable total weight, while keeping
        # the running Fisher on the raw-Fisher scale. (Normalizing to unit mean
        # -- scale = n_elem / total -- multiplied the Fisher by 1/mean(F) ~
        # 1e4-1e5, making --ewc_lambda that many times too strong for online EWC
        # and collapsing the accuracy-matrix diagonal.)
        task_mean = total / n_elem
        if self._ref_mean is None:
            self._ref_mean = task_mean
        scale = self._ref_mean / task_mean
        return {n: f * scale for n, f in fisher.items()}

    def consolidate(self, model, tokenizer, texts: List[str], device: str,
                    max_seq_len: int = 256, n_samples: int = 200) -> None:
        task_ewc = _FisherEstimate(
            model, tokenizer, texts, device,
            max_seq_len=max_seq_len,
            n_samples=n_samples,
        )
        task_fisher = self._normalize(task_ewc.fisher)

        new_fisher: Dict[str, torch.Tensor] = {
            name: self.gamma * fisher.float()
            for name, fisher in self.fisher.items()
        }
        for name, fisher in task_fisher.items():
            current = fisher.detach().float().cpu()
            if name in self.fisher:
                new_fisher[name] = new_fisher[name] + current
            else:
                new_fisher[name] = current.clone()
        self.fisher = new_fisher

        new_params = {
            name: param.detach().float().cpu()
            for name, param in task_ewc.params.items()
        }
        for name, ref in self.params.items():
            if name not in new_params:
                new_params[name] = ref
        self.params = new_params
        self.n_tasks += 1

    def penalty(self, model) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        if not self.fisher:
            return loss

        params = dict(model.named_parameters())
        for name, fisher in self.fisher.items():
            p = params.get(name)
            ref = self.params.get(name)
            if p is None or ref is None:
                continue
            # C1 fix: the running Fisher self.fisher already encodes the gamma
            # recursion (consolidate: F~_t = gamma*F~_{t-1} + F_t). Multiplying
            # by gamma again here double-decays it. Use the accumulated Fisher
            # directly (Schwarz 2018 / Huszar 2018 penalty = (lambda/2) F~_t (.)^2).
            f = fisher.to(p.device)
            loss = loss + (f * (p.float() - ref.to(p.device)).pow(2)).sum()
        return loss
