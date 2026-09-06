"""Task-boundary checkpoint state for resumable continual-learning runs.

The running state includes RNG streams, trainable parameters, online-EWC and
SI payloads, timing, and method-specific state. Sequential OSRM stores one
immutable feature dictionary per completed task. O-LoRA reconstructs its
accumulated factors from the already-written per-task checkpoints.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from typing import Dict, List, Optional

import numpy as np
import torch

RESUME_DIR = "resume"
_FORMAT_VERSION = 1

# Args that may differ between the original and the resuming invocation
# without changing the training trajectory. The resume-control flags
# (resume_from/start_task/keep_resume) are how a continuation is LAUNCHED, not
# part of the trajectory, so the writer and reader legitimately differ on them.
_SIG_EXCLUDE = ("output_dir", "no_mem_log", "resume",
                "resume_from", "start_task", "keep_resume")


# ── paths / io ───────────────────────────────────────────────────────────

def state_path(output_dir: str) -> str:
    return os.path.join(output_dir, RESUME_DIR, "state.pt")


def task_aux_path(output_dir: str, t: int) -> str:
    return os.path.join(output_dir, RESUME_DIR, f"task_{t}.pt")


def atomic_save(obj, path: str) -> None:
    """torch.save via tmp + os.replace so a kill never leaves a torn file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def clear(output_dir: str) -> None:
    """Delete the resume dir. Removes state.pt FIRST so the deletion is
    crash-atomic: a kill mid-rmtree then leaves only orphan task_<t>.pt files,
    which probe() ignores (no state.pt -> returns None -> clean restart),
    rather than a state.pt pointing at already-deleted task aux files."""
    d = os.path.join(output_dir, RESUME_DIR)
    try:
        os.remove(os.path.join(d, "state.pt"))
    except FileNotFoundError:
        pass
    shutil.rmtree(d, ignore_errors=True)


# ── signature ────────────────────────────────────────────────────────────

def build_signature(args, n_tasks_effective: int) -> dict:
    sig = {k: v for k, v in vars(args).items() if k not in _SIG_EXCLUDE}
    sig["__n_tasks_effective"] = int(n_tasks_effective)
    sig["__format_version"] = _FORMAT_VERSION
    # Pin the device: train.py picks the param dtype purely from it
    # (bf16-on-cuda vs fp32-on-cpu), so a GPU-kill / CPU-resume (or the
    # reverse) would otherwise pass the signature check and silently train in
    # a different dtype. probe() then raises a clear saved=cuda/now=cpu error.
    sig["__device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return sig


def probe(output_dir: str, signature: dict,
          ignore_sig_keys: tuple = ()) -> Optional[dict]:
    """Return the saved resume state if one exists, else None.

    A signature mismatch raises: resuming a DIFFERENT configuration into the
    same output_dir is almost certainly an accident, and silently restarting
    from task 0 is exactly what resume exists to prevent.

    `ignore_sig_keys` skips keys in the comparison — used by TSH
    promotion continuation to ignore the task horizon when the prior run is a
    legitimate shorter prefix.
    """
    path = state_path(output_dir)
    if not os.path.exists(path):
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # torn/corrupt file: warn loudly, start fresh
        print(f"  Resume: WARNING — could not read {path} ({e}); "
              "starting from task 0.")
        return None
    saved_sig = state.get("signature", {})
    keys = (set(saved_sig) | set(signature)) - set(ignore_sig_keys)
    diffs = sorted(k for k in keys if saved_sig.get(k) != signature.get(k))
    if diffs:
        details = "; ".join(
            f"{k}: saved={saved_sig.get(k)!r} now={signature.get(k)!r}"
            for k in diffs)
        raise RuntimeError(
            f"{path} was written by a run with different settings "
            f"({details}). Refusing to mix runs: rerun with the original "
            "settings to continue it, or pass --no-resume to restart this "
            "configuration from task 0.")
    return state


# ── RNG ──────────────────────────────────────────────────────────────────

def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else []),
    }


def restore_rng(state: dict) -> None:
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    cuda_states = rng.get("torch_cuda") or []
    if torch.cuda.is_available() and cuda_states:
        n = torch.cuda.device_count()
        if len(cuda_states) == n:
            torch.cuda.set_rng_state_all(cuda_states)
        else:  # different visible-GPU count on the resuming node
            print(f"  Resume: WARNING — saved {len(cuda_states)} CUDA RNG "
                  f"state(s) but the resuming node has {n} visible GPU(s); "
                  "restoring the overlap only. Replay/generation RNG may "
                  "diverge from the original run.")
            for i in range(min(n, len(cuda_states))):
                torch.cuda.set_rng_state(cuda_states[i], i)
    elif torch.cuda.is_available() and not cuda_states:
        print("  Resume: WARNING — no saved CUDA RNG state but CUDA is "
              "available now (the run was killed on CPU?); CUDA RNG starts "
              "from the seed, which may diverge replay/generation.")


# ── trainable params ─────────────────────────────────────────────────────

def collect_trainable(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone().cpu()
            for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, saved: Dict[str, torch.Tensor]) -> int:
    """Copy saved values over the freshly built model's params, strictly.

    Every saved name must exist (same shape) and every currently-trainable
    param must be covered; a partial restore would continue from a
    half-initialized adapter.
    """
    own = dict(model.named_parameters())
    missing = sorted(n for n in saved if n not in own)
    if missing:
        raise RuntimeError(
            f"resume: {len(missing)} saved params not found in the rebuilt "
            f"model (first: {missing[:3]}) — model construction diverged "
            "from the original run.")
    with torch.no_grad():
        for n, t in saved.items():
            p = own[n]
            if tuple(p.shape) != tuple(t.shape):
                raise RuntimeError(
                    f"resume: shape mismatch for {n}: saved "
                    f"{tuple(t.shape)} vs model {tuple(p.shape)}.")
            p.data.copy_(t.to(device=p.device, dtype=p.dtype))
    uncovered = sorted(
        n for n, p in own.items()
        if p.requires_grad and n not in saved)
    if uncovered:
        raise RuntimeError(
            f"resume: {len(uncovered)} trainable params have no saved value "
            f"(first: {uncovered[:3]}) — refusing a partial restore.")
    return len(saved)


# ── method-state payloads ────────────────────────────────────────────────

def _to_cpu(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in d.items()}


def _to_device(d: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in d.items()}


def online_ewc_payload(oe) -> dict:
    return {
        "gamma": oe.gamma,
        "normalize_fisher": oe.normalize_fisher,
        "n_tasks": oe.n_tasks,
        "ref_mean": oe._ref_mean,
        "fisher": _to_cpu(oe.fisher),
        "params": _to_cpu(oe.params),
    }


def online_ewc_from_payload(payload: dict):
    """Rebuild OnlineEWC. Tensors stay on CPU — consolidate() accumulates
    new task Fishers on CPU, so a GPU-restored running Fisher would crash
    at the first post-resume consolidate."""
    from .ewc import OnlineEWC
    oe = OnlineEWC(gamma=payload["gamma"],
                   normalize_fisher=payload["normalize_fisher"])
    oe.fisher = dict(payload["fisher"])
    oe.params = dict(payload["params"])
    oe.n_tasks = int(payload["n_tasks"])
    oe._ref_mean = payload["ref_mean"]
    return oe


def si_payload(si) -> dict:
    # At a task boundary consolidate() has already folded the path integral:
    # _task_start/_omega/_step_task_grads are empty, only the cumulative
    # importance + anchors persist.
    return {"importance": _to_cpu(si.importance), "params": _to_cpu(si.params)}


def si_restore(si, payload: dict, device) -> None:
    si.importance = _to_device(payload["importance"], device)
    si.params = _to_device(payload["params"], device)


# ── per-task aux files (immutable, append-only) ──────────────────────────

def save_task_aux(output_dir: str, t: int, aux: dict) -> None:
    if aux:
        atomic_save(aux, task_aux_path(output_dir, t))


def load_task_aux(output_dir: str, t: int) -> dict:
    path = task_aux_path(output_dir, t)
    if not os.path.exists(path):
        raise RuntimeError(
            f"resume: missing {path} — the per-task state files under "
            f"{os.path.join(output_dir, RESUME_DIR)} are required to resume "
            "this method; if they were deleted, restart with --no-resume.")
    return torch.load(path, map_location="cpu", weights_only=False)


# ── running state ────────────────────────────────────────────────────────

def save_state(output_dir: str, signature: dict, next_task: int,
               task_train_secs: List[float],
               trainable: Dict[str, torch.Tensor],
               extra: Optional[dict] = None) -> None:
    state = {
        "signature": signature,
        "next_task": int(next_task),
        "task_train_secs": [float(s) for s in task_train_secs],
        "rng": capture_rng(),
        "trainable": trainable,
    }
    if extra:
        state.update(extra)
    atomic_save(state, state_path(output_dir))


# ── reconstruction helpers ───────────────────────────────────────────────

def olora_rebuild_accumulated(model, output_dir: str,
                              upto_task_exclusive: int) -> int:
    """Re-concatenate tasks 0..upto-1 loranew factors into the accumulated
    frozen lora_A/lora_B buffers, in fold order.

    Mirrors fold_current_into_prior's concat exactly (the on-disk factors are
    byte-identical to what was folded: save_olora_loranew runs immediately
    before the fold) WITHOUT calling the fold itself — fold's loranew re-init
    draws from the RNG, and the live post-fold loranew values are restored
    separately from the trainable payload.
    """
    from safetensors.torch import load_file
    from .olora_impl import OLoRALinear
    mods = {name: m for name, m in model.named_modules()
            if isinstance(m, OLoRALinear)}
    if not mods:
        raise RuntimeError("resume: no OLoRALinear modules found to rebuild.")
    for s in range(upto_task_exclusive):
        fpath = os.path.join(output_dir, f"after_task_{s}",
                             "olora_loranew.safetensors")
        if not os.path.exists(fpath):
            raise RuntimeError(
                f"resume: missing {fpath} — O-LoRA resume rebuilds the "
                "accumulated adapter from the per-task factor files; if "
                "checkpoints were pruned mid-run, restart with --no-resume.")
        fac = load_file(fpath)
        with torch.no_grad():
            for name, m in mods.items():
                A = fac[name + ".A"].to(m.lora_A.dtype).to(m.lora_A.device)
                B = fac[name + ".B"].to(m.lora_B.dtype).to(m.lora_B.device)
                m.lora_A = torch.cat([m.lora_A, A], dim=0)
                m.lora_B = torch.cat([m.lora_B, B], dim=1)
    return len(mods)
