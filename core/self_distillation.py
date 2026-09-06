from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_distill_loss(student_logits: torch.Tensor,
                    teacher_logits: torch.Tensor,
                    attention_mask: torch.Tensor,
                    temperature: float = 2.0) -> torch.Tensor:
    """LwF distillation loss between next-token student/teacher logits.

    Shared by both the PEFT-adapter teacher (:class:`SelfDistillation`) and the
    wrapper-param-snapshot teacher (:class:`FrozenStateTeacher`); the only
    difference between the two is HOW the teacher logits are produced. Masks the
    last position (no next token) and multiplies by T² so the gradient scale is
    independent of temperature (Hinton et al.)."""
    T = temperature
    s_logits = student_logits[:, :-1, :] / T
    t_logits = teacher_logits[:, :-1, :] / T
    mask = attention_mask[:, 1:].float()

    t_probs = F.softmax(t_logits, dim=-1)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)
    # Full-vocabulary forward KL: KL(teacher || student).
    kl_per_token = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)

    kl_masked = (kl_per_token * mask).sum() / mask.sum().clamp(min=1.0)
    return kl_masked * (T ** 2)


class SelfDistillation:
    """
    Memory-efficient self-distillation using PEFT's multi-adapter support.
    Instead of deep-copying the entire model, adds the current adapter as a
    frozen "teacher" adapter and switches between teacher/student via
    set_adapter() during training.
    """

    TEACHER = "_sd_teacher"

    def __init__(self, model, device: str, temperature: float = 2.0):
        self.temperature = temperature
        self.device = device
        self.model_ref = model

        # Create a teacher adapter by cloning the current (default) adapter
        model.add_adapter(self.TEACHER, model.peft_config["default"])

        # Copy default adapter weights into the teacher adapter
        all_params = dict(model.named_parameters())
        with torch.no_grad():
            for name, param in all_params.items():
                if f".{self.TEACHER}." in name:
                    src_name = name.replace(f".{self.TEACHER}.", ".default.")
                    if src_name in all_params:
                        param.data.copy_(all_params[src_name].data)

        # Switch back to student (default) adapter
        model.set_adapter("default")

        # Explicitly freeze the teacher copy. PEFT's set_adapter("default")
        # already leaves non-active adapters with requires_grad=False
        # (verified on peft 0.19), but the task optimizer is built AFTER this
        # constructor, so don't let the invariant rest on a PEFT
        # implementation detail — an accidentally-trainable teacher would
        # silently land in the optimizer param groups.
        for name, p in model.named_parameters():
            if f".{self.TEACHER}." in name:
                p.requires_grad_(False)

    def distill_loss(self, student_logits: torch.Tensor,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     model=None) -> torch.Tensor:
        m = model or self.model_ref

        # Switch to frozen teacher adapter for the forward pass
        was_training = m.training
        m.set_adapter(self.TEACHER)
        m.eval()
        with torch.no_grad():
            teacher_logits = m(input_ids=input_ids,
                               attention_mask=attention_mask).logits
        m.set_adapter("default")
        if was_training:
            m.train()
        else:
            m.eval()

        return kl_distill_loss(student_logits, teacher_logits, attention_mask,
                               self.temperature)

    def cleanup(self):
        self.model_ref.set_adapter("default")
        self.model_ref.delete_adapter(self.TEACHER)
        self.model_ref = None
        torch.cuda.empty_cache()


class FrozenStateTeacher:
    """LwF teacher for custom-wrapper CL methods (currently O-LoRA).

    Wrapper methods replace target Linears with their own ``nn.Module`` wrappers
    and train on a *plain* model (not a ``PeftModel``), so
    :class:`SelfDistillation`'s ``add_adapter``/``set_adapter`` machinery does
    not apply. Instead this teacher snapshots every trainable parameter at the
    START of a task (i.e. the end-of-previous-task state — the freshly-created
    current-task adapter is zero-init and contributes nothing yet, so the
    snapshot reproduces the model after tasks ``0..t-1`` exactly). The teacher
    forward temporarily swaps the snapshot into the live parameters' ``.data``,
    runs under ``no_grad``, and restores the live values before returning, so
    the student's backward pass (run later by the caller) sees the up-to-date
    weights.

    Drop-in for the ``distiller`` used by ``train_one_task`` (same
    ``distill_loss(student_logits, input_ids, attention_mask, model)`` signature).
    Construct only at ``t > 0`` (no prior task to distill toward at task 0).
    """

    def __init__(self, model, device: str, temperature: float = 2.0):
        self.temperature = temperature
        self.device = device
        self.model_ref = model
        # Snapshot the task-start state of every trainable param (= prev-task model).
        self.snapshot = {n: p.detach().clone()
                         for n, p in model.named_parameters() if p.requires_grad}

    def distill_loss(self, student_logits: torch.Tensor,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     model=None) -> torch.Tensor:
        m = model or self.model_ref
        # Swap snapshot -> live params by .data REBINDING — deliberately NOT
        # p.copy_(snap): distill_loss runs between the student's forward and
        # backward, and the student's autograd graph holds the live param
        # tensors as saved-for-backward. An in-place copy_ bumps their
        # version counters (even under no_grad) and the student backward
        # then fails with "modified by an inplace operation" (verified).
        # Rebinding leaves the saved tensors untouched. The try/finally
        # guarantees the live tensors are rebound (and train mode restored)
        # even if the teacher forward raises (e.g. OOM).
        live = {}
        was_training = m.training
        try:
            with torch.no_grad():
                for n, p in m.named_parameters():
                    snap = self.snapshot.get(n)
                    if snap is not None:
                        live[n] = p.data
                        p.data = snap
                m.eval()
                teacher_logits = m(input_ids=input_ids,
                                   attention_mask=attention_mask).logits
        finally:
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if n in live:
                        p.data = live[n]
            if was_training:
                m.train()
        return kl_distill_loss(student_logits, teacher_logits, attention_mask,
                               self.temperature)

    def cleanup(self):
        self.snapshot = None
        self.model_ref = None
        torch.cuda.empty_cache()
