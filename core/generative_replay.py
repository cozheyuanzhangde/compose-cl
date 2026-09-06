from __future__ import annotations

from typing import List, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm


class GenerativeReplay:
    """
    Generative replay for continual learning in LLMs.

    The LLM serves as both generator (producing replay samples from past
    task distributions) and solver (learning new tasks while retaining past
    knowledge through distillation on replayed samples).

    At each task boundary the current adapter weights are snapshotted into a
    frozen "teacher" adapter (same PEFT multi-adapter trick used by
    SelfDistillation).  The teacher then:
      1. Generates replay texts unconditionally from a seed token.
      2. Provides soft targets for the student on those replay texts via
         forward-KL distillation.

    Solver loss:
        L = (1 - w) * CE(current data) + w * KL(teacher || student on replay)
    where w is the replay weight (controlled by --replay_weight); 1-w weights CE.

    Returned replay samples are structured as:
        {
            "prompt": "...",
            "answer": " ..."
        }
    so downstream training can mask prompt tokens cleanly.
    """

    TEACHER = "_gr_teacher"

    def __init__(self, model, device: str, temperature: float,
                 replay_token: str):
        self.device = device
        self.temperature = temperature
        self.model_ref = model
        self.replay_token = replay_token

        # Snapshot current adapter as frozen teacher
        model.add_adapter(self.TEACHER, model.peft_config["default"])
        all_params = dict(model.named_parameters())
        with torch.no_grad():
            for name, param in all_params.items():
                if f".{self.TEACHER}." in name:
                    src_name = name.replace(f".{self.TEACHER}.", ".default.")
                    if src_name in all_params:
                        param.data.copy_(all_params[src_name].data)

        # Explicitly freeze teacher adapter params
        for name, param in model.named_parameters():
            if f".{self.TEACHER}." in name:
                param.requires_grad = False

        model.set_adapter("default")

    @torch.inference_mode()
    def generate_replay_texts(
        self,
        model,
        tokenizer,
        n_samples: int,
        max_new_tokens: int = 128,
        gen_temperature: float = 0.7,
        batch_size: int = 32,
    ) -> List[Dict[str, str]]:
        """Generate structured replay samples using the teacher adapter.

        Replay is unconditional: the teacher generates from one shared frozen
        replay token. No past data or task identifier is supplied.

        Returns:
            List[{"prompt": ..., "answer": ...}]
        """
        model.set_adapter(self.TEACHER)
        model.eval()

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id
        prev_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        results = self._generate_unconditional(
            model, tokenizer, n_samples,
            max_new_tokens, gen_temperature, batch_size,
            pad_id, eos_id,
        )

        tokenizer.padding_side = prev_padding_side
        model.set_adapter("default")
        model.train()
        return results

    # ------------------------------------------------------------------
    # Unconditional replay (strict CL — no past data stored)
    # ------------------------------------------------------------------
    def _generate_unconditional(
        self, model, tokenizer,
        n_samples: int,
        max_new_tokens: int, gen_temperature: float, batch_size: int,
        pad_id: int, eos_id: int,
    ) -> List[Dict[str, str]]:
        bos_id = tokenizer.convert_tokens_to_ids(self.replay_token)
        replay_text = self.replay_token

        replay_items: List[Dict[str, str]] = []

        for start in tqdm(range(0, n_samples, batch_size),
                          desc="    Generating replay (unconditional)",
                          leave=False):
            cur_batch = min(batch_size, n_samples - start)

            # Identical seed for every sample in the batch — diversity
            # comes entirely from sampling randomness.
            input_ids = torch.tensor(
                [[bos_id]] * cur_batch, dtype=torch.long, device=model.device)
            attention_mask = torch.ones_like(input_ids)

            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=gen_temperature if gen_temperature > 0 else 1.0,
                do_sample=gen_temperature > 0,
                top_p=0.9,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

            for i in range(cur_batch):
                gen_ids = out_ids[i, 1:]  # skip the BOS seed token

                stop_mask = (gen_ids == eos_id) | (gen_ids == pad_id)
                stop_pos = stop_mask.nonzero(as_tuple=True)[0]
                if len(stop_pos) > 0:
                    gen_ids = gen_ids[:stop_pos[0]]

                gen_text = tokenizer.decode(
                    gen_ids, skip_special_tokens=True).strip()

                if not gen_text:
                    continue

                # Prompt is the minimal seed; answer is the full generation.
                # This means answer-only masking trains on nearly all tokens.
                replay_items.append({
                    "prompt": replay_text,
                    "answer": gen_text,
                })

        return replay_items

    def replay_distill_loss(
        self,
        student_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        model=None,
    ) -> torch.Tensor:
        """Forward-KL distillation loss on replay data, answer tokens only."""
        m = model or self.model_ref

        m.set_adapter(self.TEACHER)
        m.eval()
        with torch.no_grad():
            teacher_logits = m(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        m.set_adapter("default")
        m.train()

        T = self.temperature
        s_logits = student_logits[:, :-1, :] / T
        t_logits = teacher_logits[:, :-1, :] / T

        # logits[:, t] predicts token at position t+1
        token_mask = loss_mask[:, 1:].float()

        t_probs = F.softmax(t_logits, dim=-1)
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        t_log_probs = F.log_softmax(t_logits, dim=-1)

        kl_per_token = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
        kl_masked = (kl_per_token * token_mask).sum() / token_mask.sum().clamp(min=1.0)
        return kl_masked * (T ** 2)

    def cleanup(self):
        self.model_ref.set_adapter("default")
        self.model_ref.delete_adapter(self.TEACHER)
        self.model_ref = None
        torch.cuda.empty_cache()


class OLoRAGenerativeReplay(GenerativeReplay):
    """Generative replay for the non-PEFT O-LoRA wrapper.

    The PEFT base class snapshots the prev-task model as a TEACHER *adapter*
    (``add_adapter``/``set_adapter``/``disable_adapter``), which the custom
    ``OLoRALinear`` modules do not support. For O-LoRA the prev-task teacher is
    simply the model with the trainable ``loranew_B`` zeroed: ``loranew_B`` is
    zero-initialised at task start (olora_impl.py), so at that moment the live
    model already *is* the prev task, and during training we recover the teacher
    by temporarily zeroing ``loranew_B`` around a forward. This reuses the
    parent's ``_generate_unconditional`` generation and KL math unchanged — only
    the teacher mechanism differs.
    """

    def __init__(self, model, device: str, temperature: float,
                 replay_token: str):
        # Deliberately DO NOT call super().__init__ (it PEFT-add_adapters a
        # TEACHER). No snapshot is needed: the accumulated frozen lora_A/B IS
        # the teacher; loranew is the only (zeroable) student delta.
        self.device = device
        self.temperature = temperature
        self.model_ref = model
        self.replay_token = replay_token

    @staticmethod
    def _zero_loranew_B(model):
        """Zero every trainable loranew_B (-> model == prev-task teacher).
        Returns the saved tensors for restoration."""
        saved = {}
        for n, p in model.named_parameters():
            if n.endswith("loranew_B"):
                saved[n] = p.data.clone()
                p.data.zero_()
        return saved

    @staticmethod
    def _restore_loranew_B(model, saved):
        for n, p in model.named_parameters():
            if n in saved:
                p.data.copy_(saved[n])

    @torch.inference_mode()
    def generate_replay_texts(self, model, tokenizer, n_samples,
                              max_new_tokens: int = 128,
                              gen_temperature: float = 0.7, batch_size: int = 32):
        # Generation runs at task start (loranew_B already 0 => live model ==
        # teacher); zero it anyway for safety and DON'T touch PEFT adapters.
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        model.eval()
        saved = self._zero_loranew_B(model)
        try:
            results = self._generate_unconditional(
                model, tokenizer, n_samples, max_new_tokens,
                gen_temperature, batch_size, pad_id, eos_id)
        finally:
            self._restore_loranew_B(model, saved)
            tokenizer.padding_side = prev_side
            model.train()
        return results

    def replay_distill_loss(self, student_logits, input_ids, attention_mask,
                            loss_mask, model=None):
        """Forward-KL on replay data, answer tokens only — teacher = prev task
        (loranew_B zeroed). Identical KL math to the PEFT parent."""
        m = model or self.model_ref
        m.eval()
        saved = self._zero_loranew_B(m)
        try:
            with torch.no_grad():
                teacher_logits = m(input_ids=input_ids,
                                   attention_mask=attention_mask).logits
        finally:
            self._restore_loranew_B(m, saved)
            m.train()

        T = self.temperature
        s_logits = student_logits[:, :-1, :] / T
        t_logits = teacher_logits[:, :-1, :] / T
        token_mask = loss_mask[:, 1:].float()
        t_probs = F.softmax(t_logits, dim=-1)
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        t_log_probs = F.log_softmax(t_logits, dim=-1)
        kl_per_token = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
        kl_masked = (kl_per_token * token_mask).sum() / token_mask.sum().clamp(min=1.0)
        return kl_masked * (T ** 2)

    def cleanup(self):
        # No TEACHER adapter to delete (we never added one).
        self.model_ref = None
        torch.cuda.empty_cache()
