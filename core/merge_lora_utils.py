"""Sequential OSRM utilities for the paper's low-rank allocation baseline.

The adaptation uses only past-task averaged input features. At the start of a
new task it initializes each fresh LoRA A matrix in the null space of those
features, retaining one vector per past task and layer.
"""

from __future__ import annotations
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────
# Sequential OSRM: data-driven orthogonal A initialisation
# ─────────────────────────────────────────────────────────────────────────

def _lora_module_keys(model) -> List[str]:
    """Return all module prefixes that own a LoRA-A parameter.

    The prefix is the dotted-path up to (but excluding) `.lora_A.`. Use
    these as canonical keys when matching modules across functions.
    """
    keys = []
    for name, _ in model.named_parameters():
        if ".lora_A." in name and name.endswith(".weight"):
            keys.append(name.split(".lora_A.")[0])
    # Deduplicate while preserving order
    return list(dict.fromkeys(keys))


def _resolve_module(model, dotted_path: str):
    """Get a submodule from a dotted-path name (no `getattr_recursive` in
    torch's public API)."""
    m = model
    for part in dotted_path.split("."):
        m = getattr(m, part)
    return m


@torch.inference_mode()
def collect_lora_input_features(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    n_samples: int = 64,
    max_seq_len: int = 256,
    batch_size: int = 4,
) -> Dict[str, torch.Tensor]:
    """Compute the mean input feature for every LoRA module on `texts`.

    For each LoRA-wrapped Linear, the "input" is the activation `x` that
    gets multiplied by A. We hook each module's `lora_A` Linear, capture
    its input, and reduce it to one d_in-dim vector per module.

    FAITHFUL to the official OSRM (illidanlab/OSRM,
    `get_pretrained_latent_features`): per forward batch the hook computes
        x.mean(dim=1).mean(dim=0)        # seq-mean THEN batch-mean
    (i.e. each *sequence* contributes equally, regardless of length, and
    padded positions ARE included — the official code does not mask). The
    per-batch vectors are summed and divided by the number of batches.

    Args:
        model: PEFT-wrapped model with LoRA attached.
        tokenizer: HF tokenizer.
        texts: training text strings (sampled from the current task).
        device: cuda / cpu.
        n_samples: number of texts to use (sampled in order).
        max_seq_len: truncate each text to this many tokens.
        batch_size: forward-pass batch size.
    Returns:
        Dict mapping LoRA module prefix → average input feature (CPU
        float32 tensor of shape (d_in,)).
    """
    keys = _lora_module_keys(model)
    if not keys:
        return {}

    # Running sum of per-batch feature vectors + count of batches seen.
    sums: Dict[str, torch.Tensor] = {}
    n_batches: Dict[str, int] = {k: 0 for k in keys}
    def make_hook(key):
        def _hook(_mod, inputs, _output):
            x = inputs[0].float()  # (batch, seq, d_in)
            v = x.mean(dim=1).mean(dim=0)  # faithful OSRM reduction; pad included
            if key not in sums:
                sums[key] = v.detach().cpu()
            else:
                sums[key] += v.detach().cpu()
            n_batches[key] += 1
        return _hook

    handles = []
    for key in keys:
        # Hook on the lora_A.default Linear: its forward(input) call
        # receives x = layer input. We capture x before A·x is computed.
        try:
            mod = _resolve_module(model, key + ".lora_A.default")
        except AttributeError:
            # Some PEFT adapters might use a different name; skip.
            continue
        handles.append(mod.register_forward_hook(make_hook(key)))

    was_training = model.training
    model.eval()
    sample = texts[:n_samples]
    try:
        for start in tqdm(range(0, len(sample), batch_size),
                          desc="    OSRM feature collection", leave=False):
            batch = sample[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_seq_len).to(device)
            _ = model(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"])
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    out = {}
    for k in keys:
        if n_batches[k] > 0:
            out[k] = sums[k] / n_batches[k]  # (d_in,) CPU float32
    return out


def osrm_orthogonal_init_A(
    model,
    past_features: List[Dict[str, torch.Tensor]],
    eps: float = 1e-12,
) -> int:
    """[SEQUENTIAL-CL ADAPTATION — not the official post-hoc OSRM]

    Re-initialise the fresh task's A matrices using OSRM's null-space rule
    from PAST-task features only, then train B from zero. This is an online
    adaptation; the *faithful* OSRM (illidanlab/OSRM) is a post-hoc, all-task
    analytical merge — see `osrm_post_hoc_factorize` below. Difference: the
    official keeps A orthonormal and solves B = ΔW·pinv(A) from the trained
    delta; here we rescale A to Kaiming norm and learn B by SGD.

    For each LoRA module with prefix `key`:
        H_past = stack([past_features[i][key] for i in 0..t-1])      # (t, d_in)
        S = H_past^T @ H_past                                         # (d_in, d_in) PSD
        V, Λ = eig(S)            # descending eigenvalues
        A_new = V[:, n-r:n]^T    # smallest-r eigenvectors (rows of A)

    A_new's rows are orthonormal AND span the directions of input space
    where past-task averaged features have *least* variance. So for any
    past-task input `x ≈ Σ_i c_i · h̄_i`, `A_new @ x` is small — task
    t+1's update doesn't strongly shift past tasks' outputs.

    Because past_features has only t ≤ N rows (a few per task), S is
    rank-deficient and the bottom (d_in − t) eigenvalues are all 0.
    Picking r out of those zeros means we get *exact* orthogonality to
    the past averages.

    B remains 0 (PEFT default), so the initial ΔW = 0 — no behaviour
    shift at init; only the gradient trajectory is biased.

    Returns the number of modules updated.
    """
    if not past_features:
        return 0
    n_modules = 0
    for name, p in model.named_parameters():
        if ".lora_A." not in name or not name.endswith(".weight"):
            continue
        key = name.split(".lora_A.")[0]
        # Stack past averaged features that exist for this module
        past_h = [d[key] for d in past_features if key in d]
        if not past_h:
            continue
        H = torch.stack([h.to(p.device).float() for h in past_h], dim=0)
        # SVD on H: H = U Σ V^T, V ∈ (d_in, k). The smallest-r right-
        # singular vectors are in V[:, k:] (which we don't get from a
        # thin SVD), so we use eigendecomp of H^T H directly when d_in
        # is small enough, or use the full SVD of H with full_matrices=True.
        # H is (t, d_in) with t ≪ d_in, so full SVD returns V ∈ (d_in, d_in).
        try:
            U_, S_, Vh = torch.linalg.svd(H, full_matrices=True)
            # Vh is (d_in, d_in); its rows are right-singular vectors,
            # ordered by descending singular value. We want the smallest
            # r — these are the rows at the END of Vh (zero singular
            # values for rank-deficient H, perfectly orthogonal to row(H)).
            r = p.shape[0]  # rank of A
            d_in = p.shape[1]
            # Smallest-r right-singular vectors: Vh[-r:] are rows of d_in size.
            A_new = Vh[-r:].contiguous()  # (r, d_in), rows orthonormal
        except Exception:
            continue
        with torch.no_grad():
            # Preserve the original row-norm scale (Kaiming-init expects
            # ‖A_row‖ ≈ √(2/d_in) on average; orthonormal rows have norm
            # 1, so we scale to match).
            orig_norms = p.data.float().norm(dim=1, keepdim=True).mean()
            A_scaled = A_new * orig_norms.clamp_min(eps)
            p.data.copy_(A_scaled.to(p.dtype))
        n_modules += 1
    return n_modules


# ─────────────────────────────────────────────────────────────────────────
