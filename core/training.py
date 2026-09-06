from __future__ import annotations
import math
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .datasets_local import CLMDataset, collate_pad
from .ewc import OnlineEWC
from .mem_log import get_mem_log
from .self_distillation import SelfDistillation
from .generative_replay import GenerativeReplay
from .si import SynapticIntelligence


def _maybe_warmup_scheduler(opt, args, n_batches: int):
    """Linear warmup → constant LR. No-op (returns None) when frac<=0.

    The ramp restarts at every task boundary because each task receives a
    fresh optimizer and scheduler.

    The first optimizer step deliberately runs at lr/W, not 0 (W = warmup
    steps, ramp (step+1)/W reaching full lr exactly at step W): a 0-LR
    first step would be a pure no-op — AdamW scales both the update and
    its decoupled weight decay by lr — i.e. a wasted step. If
    frac*total_steps < 2, W clamps to 1 and warmup is effectively off.
    """
    frac = getattr(args, "warmup_frac", 0.0) or 0.0
    if frac <= 0:
        return None
    total_opt_steps = args.epochs * math.ceil(n_batches / args.grad_accum)
    warmup_steps = max(1, int(frac * total_opt_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps          # linear warmup to full LR
        return 1.0

    return LambdaLR(opt, lr_lambda)


_NO_DECAY_KEYS = ("embed_tokens", "lm_head", "norm", ".bias", "_bias")


def _build_param_groups(model, weight_decay: float):
    """Split trainable params into (decay, no-decay) groups for AdamW."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in _NO_DECAY_KEYS):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _amp_setup(model, device):
    """Configure autocast + GradScaler based on the model's parameter dtype.

    - bf16 model: autocast(bf16), scaler DISABLED. bf16 has wide exponent
      range so loss scaling is unnecessary, and GradScaler.unscale_ calls
      a CUDA kernel that isn't implemented for bf16 gradients.
    - fp16 model: autocast(fp16), scaler ENABLED (needs loss scaling).
    - fp32 / cpu: autocast disabled, scaler disabled.

    The scaler is always returned (possibly disabled) so the same
    scaler.scale/.unscale_/.step/.update call sites work unchanged.
    """
    if device != "cuda":
        return False, None, torch.amp.GradScaler("cuda", enabled=False)
    dtype = next(model.parameters()).dtype
    if dtype == torch.float16:
        return True, torch.float16, torch.amp.GradScaler("cuda", enabled=True)
    if dtype == torch.bfloat16:
        return True, torch.bfloat16, torch.amp.GradScaler("cuda", enabled=False)
    return False, None, torch.amp.GradScaler("cuda", enabled=False)


def _optimizer_step(model, opt, scaler, sched=None,
                    si_state: Optional[SynapticIntelligence] = None,
                    grad_scale: float = 1.0):
    """Apply one optimizer step and update CL gradient/trajectory state.

    grad_scale: compensation for a PARTIAL accumulation window (the
    epoch-end flush). Every micro-batch loss is pre-divided by the full
    grad_accum, so a window holding only ``rem`` micro-batches accumulates
    rem/grad_accum of a true window average; the flush passes
    grad_scale=grad_accum/rem to rescale the accumulated grads (and SI's
    pending task grads) back to a full-size average before clipping /
    stepping. 1.0 (no-op) for full windows. A constant rescale commutes
    with GradScaler.unscale_, so applying it pre-unscale is exact.
    """
    if grad_scale != 1.0:
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(grad_scale)
        if si_state is not None:
            si_state.scale_pending_grads(grad_scale)
    scaler.unscale_(opt)
    si_step = si_state.capture_step(model) if si_state is not None else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    if si_state is not None:
        si_state.update_omega(model, si_step)
    scaler.update()
    if sched is not None:
        sched.step()
    opt.zero_grad(set_to_none=True)


def train_one_task(model, tokenizer, texts, device, args,
                   ewc_terms: Optional[OnlineEWC] = None,
                   distiller: Optional[SelfDistillation] = None,
                   si_state: Optional[SynapticIntelligence] = None):
    get_mem_log().task_train_start()
    model.train()
    ds = CLMDataset(tokenizer, texts, args.max_seq_len)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate_pad(b, pad_id))
    opt = AdamW(_build_param_groups(model, args.weight_decay), lr=args.lr)
    sched = _maybe_warmup_scheduler(opt, args, len(dl))
    use_amp, amp_dtype, scaler = _amp_setup(model, device)

    ewc_merged = ewc_terms if ewc_terms else None

    for ep in range(args.epochs):
        # Per-epoch counter: the epoch-end flush below takes a partial
        # optimizer step, so the accumulation window must restart at the
        # epoch boundary — a running counter would trigger the next epoch's
        # first step after fewer than grad_accum micro-batches.
        step = 0
        total_ce, total_ewc, total_kd, total_si, n = 0.0, 0.0, 0.0, 0.0, 0
        pbar = tqdm(dl, desc=f"    Epoch {ep+1}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(**batch)
                ce_loss = out.loss / args.grad_accum
                task_loss = ce_loss
                loss = task_loss

                if si_state is not None:
                    si_state.record_task_gradients(model, task_loss)

                if ewc_merged is not None:
                    ewc_loss = args.ewc_lambda * ewc_merged.penalty(model) / args.grad_accum
                    loss = loss + ewc_loss
                    total_ewc += ewc_loss.item() * args.grad_accum

                if distiller is not None:
                    kd_loss = args.distill_alpha * distiller.distill_loss(
                        out.logits, batch["input_ids"], batch["attention_mask"],
                        model=model
                    ) / args.grad_accum
                    loss = loss + kd_loss
                    total_kd += kd_loss.item() * args.grad_accum

                if si_state is not None:
                    si_loss = args.si_lambda * si_state.penalty(model) / args.grad_accum
                    loss = loss + si_loss
                    total_si += si_loss.item() * args.grad_accum

            scaler.scale(loss).backward()
            total_ce += ce_loss.item() * args.grad_accum
            n += 1
            step += 1
            if step % args.grad_accum == 0:
                _optimizer_step(model, opt, scaler, sched=sched,
                                si_state=si_state)

            pbar.set_postfix(ce=f"{total_ce/max(n,1):.4f}")

        if step % args.grad_accum != 0:
            # Partial window (epoch tail): losses were pre-divided by the
            # full grad_accum, so rescale grads to a true window average.
            _optimizer_step(model, opt, scaler, sched=sched,
                            si_state=si_state,
                            grad_scale=args.grad_accum / (step % args.grad_accum))

        parts = [f"ce={total_ce/max(n,1):.4f}"]
        if ewc_merged:
            parts.append(f"ewc={total_ewc/max(n,1):.4f}")
        if distiller:
            parts.append(f"kd={total_kd/max(n,1):.4f}")
        if si_state is not None:
            parts.append(f"si={total_si/max(n,1):.4f}")
        print(f"    epoch {ep+1}/{args.epochs}  " + "  ".join(parts))

    model.eval()
    get_mem_log().task_train_end()


def train_one_task_olora(
    model, tokenizer, texts, device, args,
    orth_lambda: float = 0.5,
    l2_lambda: float = 0.0,
    distiller=None,
    replay=None,
    replay_texts=None,
    si_state: Optional[SynapticIntelligence] = None,
):
    """Train one task with O-LoRA's continuous orthogonality regularization
    (Wang et al., EMNLP-Findings 2023). Faithful to
    https://github.com/cmnfriend/O-LoRA/blob/main/src/uie_trainer_lora.py#L83-L108.

    Expects `model` to be wrapped with `OLoRALinear` modules from
    `olora_impl.py` (each target Linear has frozen `lora_A/B` and
    trainable `loranew_A/B`).

    Loss per micro-batch:
        loss = (CE / grad_accum)               # paper code: only CE is scaled
               + λ₁ · L_orth                   # NOT scaled by grad_accum
               + λ₂ · L_l2                     # NOT scaled by grad_accum
    where
        L_orth = Σ_layers |lora_A · loranew_A^T|_1     (L1 entry-wise)
        L_l2   = Σ_layers (‖loranew_A‖_2 + ‖loranew_B‖_2)
    """
    from .olora_impl import olora_orthogonality_loss, olora_l2_loss

    get_mem_log().task_train_start()
    model.train()
    ds = CLMDataset(tokenizer, texts, args.max_seq_len)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate_pad(b, pad_id))
    # Optional generative-replay rehearsal stream (composes with O-LoRA's
    # orth/L2/distill as an added loss term on the trainable loranew params).
    dl_replay = None
    if replay is not None and replay_texts:
        ds_replay = ReplayDataset(tokenizer, replay_texts, args.max_seq_len)
        dl_replay = DataLoader(ds_replay, batch_size=args.batch_size, shuffle=True,
                               collate_fn=lambda b: replay_collate(b, pad_id))
    w = args.replay_weight
    opt = AdamW(_build_param_groups(model, args.weight_decay), lr=args.lr)
    sched = _maybe_warmup_scheduler(opt, args, len(dl))
    use_amp, amp_dtype, scaler = _amp_setup(model, device)

    for ep in range(args.epochs):
        # Per-epoch counter: the epoch-end flush realigns the accumulation
        # window (see train_one_task for the rationale).
        step = 0
        total_ce, total_orth, total_l2, total_kd, total_rpl, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        total_si = 0.0
        replay_iter = iter(dl_replay) if dl_replay is not None else None
        pbar = tqdm(dl, desc=f"    Epoch {ep+1}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(**batch)
                ce_w = (1 - w) if dl_replay is not None else 1.0
                ce_loss = ce_w * out.loss / args.grad_accum   # CE (×(1-w) if replay)
                loss = ce_loss
                # SI records the gradients of the unregularized task objective
                # (ce_loss here, matching train_one_task's task_loss=ce_loss).
                if si_state is not None:
                    si_state.record_task_gradients(model, ce_loss)
                # Compute ortho and L2 unscaled (matches their code)
                orth_raw = olora_orthogonality_loss(model)
                if orth_lambda != 0 and orth_raw.item() != 0:
                    loss = loss + orth_lambda * orth_raw
                    total_orth += orth_raw.item()
                if l2_lambda != 0:
                    l2_raw = olora_l2_loss(model)
                    loss = loss + l2_lambda * l2_raw
                    total_l2 += l2_raw.item()
                # Optional LwF self-distillation toward the prev-task model
                # (FrozenStateTeacher). Output-space, so it composes with
                # O-LoRA's fold/reset without touching parameter identity.
                if distiller is not None:
                    kd_loss = args.distill_alpha * distiller.distill_loss(
                        out.logits, batch["input_ids"], batch["attention_mask"],
                        model=model
                    ) / args.grad_accum
                    loss = loss + kd_loss
                    total_kd += kd_loss.item() * args.grad_accum

                # Generative-replay rehearsal: answer-only kl/ce on a replay
                # batch, weighted w. Teacher = prev task (loranew_B zeroed
                # by OLoRAGenerativeReplay). Mirrors train_one_task_replay.
                if dl_replay is not None:
                    try:
                        rpl_batch = next(replay_iter)
                    except StopIteration:
                        replay_iter = iter(dl_replay)
                        rpl_batch = next(replay_iter)
                    rpl_batch = {k: v.to(device) for k, v in rpl_batch.items()}
                    rpl_out = model(input_ids=rpl_batch["input_ids"],
                                    attention_mask=rpl_batch["attention_mask"])
                    replay_raw = replay.replay_distill_loss(
                        student_logits=rpl_out.logits,
                        input_ids=rpl_batch["input_ids"],
                        attention_mask=rpl_batch["attention_mask"],
                        loss_mask=rpl_batch["loss_mask"], model=model)
                    rpl_loss = w * replay_raw / args.grad_accum
                    loss = loss + rpl_loss
                    total_rpl += replay_raw.item()

                if si_state is not None:
                    si_loss = args.si_lambda * si_state.penalty(model) / args.grad_accum
                    loss = loss + si_loss
                    total_si += si_loss.item() * args.grad_accum

            scaler.scale(loss).backward()
            total_ce += ce_loss.item() * args.grad_accum
            n += 1
            step += 1
            if step % args.grad_accum == 0:
                _optimizer_step(model, opt, scaler, sched=sched,
                                si_state=si_state)
            pbar.set_postfix(
                ce=f"{total_ce/max(n,1):.4f}",
                orth=f"{total_orth/max(n,1):.4f}",
                l2=f"{total_l2/max(n,1):.4f}")

        if step % args.grad_accum != 0:
            # Partial window (epoch tail): CE was pre-divided by the full
            # grad_accum, so rescale grads to a true window average. (The
            # unscaled orth/L2 terms keep their λ·grad_accum effective
            # weight relative to CE — the rescale preserves the ratio.)
            _optimizer_step(model, opt, scaler, sched=sched,
                            si_state=si_state,
                            grad_scale=args.grad_accum / (step % args.grad_accum))

        parts = [f"ce={total_ce/max(n,1):.4f}"]
        if total_orth > 0:
            parts.append(f"orth={total_orth/max(n,1):.4f}")
        if total_l2 > 0:
            parts.append(f"l2={total_l2/max(n,1):.4f}")
        if distiller is not None:
            parts.append(f"kd={total_kd/max(n,1):.4f}")
        if dl_replay is not None:
            parts.append(f"rpl={total_rpl/max(n,1):.4f}")
        if si_state is not None:
            parts.append(f"si={total_si/max(n,1):.4f}")
        print(f"    epoch {ep+1}/{args.epochs}  " + "  ".join(parts))

    model.eval()
    get_mem_log().task_train_end()


class ReplayDataset(Dataset):
    """Replay dataset with an answer-only KL mask."""

    def __init__(self, tokenizer, replay_items, max_seq_len: int):
        self.tokenizer = tokenizer
        self.replay_items = replay_items
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.replay_items)

    def __getitem__(self, idx):
        item = self.replay_items[idx]
        prompt = item["prompt"]
        answer = item["answer"]
        full_text = prompt + answer

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors=None,
        )
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors=None,
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))

        loss_mask = [0.0] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            loss_mask[i] = 1.0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
        }


def replay_collate(batch, pad_id: int):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    loss_mask = [x["loss_mask"] for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0)
    loss_mask = torch.nn.utils.rnn.pad_sequence(
        loss_mask, batch_first=True, padding_value=0.0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }


def train_one_task_replay(
    model, tokenizer, texts, replay_texts, device, args,
    replay: GenerativeReplay,
    ewc_terms: Optional[OnlineEWC] = None,
    distiller: Optional[SelfDistillation] = None,
    si_state: Optional[SynapticIntelligence] = None,
):
    """Train one task with answer-only teacher-to-student KL on replay text."""
    get_mem_log().task_train_start()
    model.train()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    ds_current = CLMDataset(tokenizer, texts, args.max_seq_len)
    dl_current = DataLoader(
        ds_current,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, pad_id),
    )

    ds_replay = ReplayDataset(tokenizer, replay_texts, args.max_seq_len)
    dl_replay = DataLoader(
        ds_replay,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: replay_collate(b, pad_id),
    )

    opt = AdamW(_build_param_groups(model, args.weight_decay), lr=args.lr)
    sched = _maybe_warmup_scheduler(opt, args, len(dl_current))
    use_amp, amp_dtype, scaler = _amp_setup(model, device)

    ewc_merged = ewc_terms if ewc_terms else None
    w = args.replay_weight

    for ep in range(args.epochs):
        # Per-epoch counter: the epoch-end flush realigns the accumulation
        # window (see train_one_task for the rationale).
        step = 0
        total_ce, total_rpl, total_ewc, total_kd, total_si, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        replay_iter = iter(dl_replay)
        pbar = tqdm(dl_current, desc=f"    Epoch {ep+1}/{args.epochs}", leave=False)

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            try:
                rpl_batch = next(replay_iter)
            except StopIteration:
                replay_iter = iter(dl_replay)
                rpl_batch = next(replay_iter)
            rpl_batch = {k: v.to(device) for k, v in rpl_batch.items()}

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                # Current-task CE
                out = model(**batch)
                ce_loss = (1 - w) * out.loss / args.grad_accum

                # Replay branch
                rpl_out = model(
                    input_ids=rpl_batch["input_ids"],
                    attention_mask=rpl_batch["attention_mask"],
                )
                replay_raw_loss = replay.replay_distill_loss(
                    student_logits=rpl_out.logits,
                    input_ids=rpl_batch["input_ids"],
                    attention_mask=rpl_batch["attention_mask"],
                    loss_mask=rpl_batch["loss_mask"],
                    model=model,
                )

                rpl_loss = w * replay_raw_loss / args.grad_accum
                task_loss = ce_loss + rpl_loss

                loss = task_loss

                if si_state is not None:
                    si_state.record_task_gradients(model, task_loss)

                if ewc_merged is not None:
                    ewc_loss = args.ewc_lambda * ewc_merged.penalty(model) / args.grad_accum
                    loss = loss + ewc_loss
                    total_ewc += ewc_loss.item() * args.grad_accum

                if distiller is not None:
                    kd_loss = args.distill_alpha * distiller.distill_loss(
                        out.logits, batch["input_ids"], batch["attention_mask"],
                        model=model,
                    ) / args.grad_accum
                    loss = loss + kd_loss
                    total_kd += kd_loss.item() * args.grad_accum


                if si_state is not None:
                    si_loss = args.si_lambda * si_state.penalty(model) / args.grad_accum
                    loss = loss + si_loss
                    total_si += si_loss.item() * args.grad_accum

            scaler.scale(loss).backward()
            total_ce += ce_loss.item() * args.grad_accum
            total_rpl += rpl_loss.item() * args.grad_accum
            n += 1
            step += 1

            if step % args.grad_accum == 0:
                _optimizer_step(model, opt, scaler, sched=sched,
                                si_state=si_state)

            pbar.set_postfix(
                ce=f"{total_ce/max(n,1):.4f}",
                rpl=f"{total_rpl/max(n,1):.4f}",
                mode="kl",
            )

        if step % args.grad_accum != 0:
            # Partial window (epoch tail): losses were pre-divided by the
            # full grad_accum, so rescale grads to a true window average.
            _optimizer_step(model, opt, scaler, sched=sched,
                            si_state=si_state,
                            grad_scale=args.grad_accum / (step % args.grad_accum))

        parts = [
            f"ce={total_ce/max(n,1):.4f}",
            f"replay={total_rpl/max(n,1):.4f}",
            "mode=kl",
        ]
        if ewc_merged:
            parts.append(f"ewc={total_ewc/max(n,1):.4f}")
        if distiller:
            parts.append(f"kd={total_kd/max(n,1):.4f}")
        if si_state is not None:
            parts.append(f"si={total_si/max(n,1):.4f}")
        print(f"    epoch {ep+1}/{args.epochs}  " + "  ".join(parts))

    model.eval()
    get_mem_log().task_train_end()
