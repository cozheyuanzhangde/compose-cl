"""Registration for the frozen control token used by generative replay.

The row is deterministically mean-initialized and remains frozen, matching the
data-anchor implementation used in the paper.
"""
from __future__ import annotations

import torch


# In-distribution control token used to seed unconditional generative replay.
# Prepended to every training sequence (so the model learns
# ``<seed> -> Question: ... Answer: ...``) and then used as the generation seed,
# putting replay generation on the fine-tuning distribution.
REPLAY_TOKEN = "<|replay_token|>"


def add_replay_token(
    tokenizer,
    model,
    token: str = REPLAY_TOKEN,
    verbose: bool = True,
) -> int:
    """Register a single replay control token and return its id.

    Design choices:

    - The new embedding row is initialised DETERMINISTICALLY to the mean of the
      existing token embeddings (no ``randn`` -> does not consume the global RNG
      and therefore does not perturb the data shuffle / dropout, unlike the
      separate-generator load bug noted in the replay analysis).
    - The row is left **frozen** and NOT added to ``modules_to_save``: the
      replay-token embedding only needs to be a *consistent* vector that the (LoRA) attention
      layers can learn to map onto the task distribution. This keeps the
      trainable surface LoRA-only — directly comparable to every other variant —
      and avoids bloating checkpoints with full ``embed_tokens`` / ``lm_head``
      copies.

    Idempotent and shape-safe: most models (e.g. Qwen3) pad the embedding table
    beyond the tokenizer vocab, so a single added token usually lands inside the
    existing table and needs no resize. The row stays frozen at its mean-init
    value, so the *same* vector is used at train and eval time: evaluate.py prepends
    the replay token to eval prompts when ``--replay_token`` is set (matching the
    training prefix), and the frozen row keeps that consistent without ever
    needing gradients.
    """
    before = len(tokenizer)
    tokenizer.add_tokens([token], special_tokens=True)
    tok_id = tokenizer.convert_tokens_to_ids(token)

    cur_embed = model.get_input_embeddings().weight.shape[0]
    if tok_id >= cur_embed:
        model.resize_token_embeddings(len(tokenizer))
        if verbose:
            print(f"  Resized embed table up for replay token: "
                  f"{cur_embed} -> {len(tokenizer)}")

    with torch.no_grad():
        input_embs = model.get_input_embeddings().weight
        input_embs[tok_id] = input_embs[:before].mean(dim=0)
        out_embs = model.get_output_embeddings()
        if out_embs is not None and out_embs.weight.data_ptr() != input_embs.data_ptr():
            out_w = out_embs.weight
            out_w[tok_id] = out_w[:before].mean(dim=0)

    if verbose:
        print(f"  Replay token {token!r} -> id {tok_id} "
              f"(mean-init, frozen, LoRA-only).")
    return tok_id
