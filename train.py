from __future__ import annotations
import json
import os
import argparse
import random
import sys
import time
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from core.datasets_local import box_answer_texts
from core.ewc import OnlineEWC
from core.self_distillation import SelfDistillation
from core.si import SynapticIntelligence
from core.generative_replay import (
    GenerativeReplay, OLoRAGenerativeReplay)
from core.training import (
    train_one_task, train_one_task_replay, train_one_task_olora,
)
from core.replay_token import add_replay_token, REPLAY_TOKEN
from core.paths import seed_dir, task_order
from core.io_utils import drop_page_cache
from core import resume as resume_io
from core.merge_lora_utils import (
    collect_lora_input_features,
    osrm_orthogonal_init_A,
)


def _save_train_timing(output_dir: str, n_tasks: int,
                       task_train_secs: List[float]) -> None:
    """Persist wall-clock training time for downstream aggregation.

    Written to <output_dir>/train_timing.json (the checkpoint dir during
    training); evaluate.py reads it back from --checkpoint_dir and folds
    total_train_time_sec / avg_train_time_per_task_sec into results.json so
    multi-seed runs can be averaged. `task_train_secs` holds the duration of
    each train-one-task optimization call. The average is always total / n_tasks.
    """
    total = float(sum(task_train_secs))
    timing = {
        "total_train_time_sec": total,
        "n_tasks": n_tasks,
        "avg_train_time_per_task_sec": total / n_tasks if n_tasks else 0.0,
        "per_task_train_time_sec": [float(s) for s in task_train_secs],
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(f"\nTrain timing: total {total:.1f}s over {n_tasks} task(s) "
          f"(avg {timing['avg_train_time_per_task_sec']:.1f}s/task) -> "
          f"{os.path.join(output_dir, 'train_timing.json')}")


def _save_train_config(output_dir: str, args) -> None:
    """Record ALL training hyperparameters to <output_dir>/train_config.json (the
    checkpoint dir). evaluate.py reads it back and folds it into the results dir's
    hyperparameters.json, so every run's results carry the full train+eval HP set.
    Written once up front, so the config is preserved even if training is later killed."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=str,
                        default="data/synthetic_qa/symbol_qa")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_tasks", type=int, default=0,
                        help="If >0, train on only the first K tasks of the "
                             "manifest (default 0 = all tasks). Lets you run a "
                             "shorter task stream without regenerating data.")
    parser.add_argument("--task_order_seed", type=int, default=None,
                        help="If set, PERMUTE the task stream order by this seed "
                             "(stream position i reads on-disk task perm[i]); "
                             "default None = canonical on-disk order 0..N-1 "
                             "(behaviour unchanged). The permutation is built over "
                             "the full manifest horizon and sliced, so it is "
                             "prefix-stable across TSH rungs. Use a fixed dev seed "
                             "to tune HPs on a held-out ordering, then report on the "
                             "canonical order by omitting the flag. Enters the resume "
                             "signature, so a canonical checkpoint is never reused "
                             "for a shuffled run.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay (paper setting: 0.01).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Micro-batch size. Effective batch = batch_size * "
                             "grad_accum (default 8*1 = 8). At fixed effective "
                             "batch, batch_size vs grad_accum is a speed/VRAM "
                             "tradeoff, not a learning one.")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient-accumulation steps (effective batch = "
                             "batch_size * grad_accum).")
    parser.add_argument("--max_seq_len", type=int, default=384,
                        help="Truncation cap for training sequences (padding is "
                             "dynamic to the batch max, so a larger cap costs "
                             "nothing on short data). 384 covers the longest "
                             "real_qa items (e.g. medmcqa vignettes ~280 tok); "
                             "--replay_max_tokens defaults to this.")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)   # 2*lora_r
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated LoRA target_modules. Default is "
                             "the 'all-linear' set (attn q/k/v/o + MLP "
                             "gate/up/down).")
    # Online EWC weight anchor
    parser.add_argument("--ewc_lambda", type=float, default=1000.0,
                        help="Online-EWC penalty strength (paper setting: 1000).")
    parser.add_argument("--ewc_n_samples", type=int, default=1000,
                        help="Samples for Fisher estimation")
    parser.add_argument("--online_EWC", "--online_ewc", dest="online_EWC",
                        action="store_true",
                        help="Enable online EWC. Stores one running Fisher "
                             "and one latest parameter anchor instead of a "
                             "per-task EWC list.")
    parser.add_argument("--online_ewc_gamma", type=float, default=1.0,
                        help="Online EWC forgetting factor gamma in [0, 1]. "
                             "The running Fisher update is "
                             "F*_t = gamma * F*_{t-1} + F_t.")
    parser.add_argument("--online_ewc_normalize_fisher",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Rescale every task Fisher to the first task's mean "
                             "before online accumulation. On by default; pass "
                             "--no-online_ewc_normalize_fisher to use raw Fishers.")
    # Synaptic Intelligence
    parser.add_argument("--SI", action="store_true",
                        help="Enable Synaptic Intelligence (Zenke et al., "
                             "ICML 2017). Tracks online path-integral "
                             "importance for trainable parameters; no old "
                             "samples are stored.")
    parser.add_argument("--si_lambda", type=float, default=1,
                        help="SI surrogate-loss strength (paper setting: 1).")
    parser.add_argument("--si_xi", type=float, default=0.1,
                        help="SI damping term xi in Omega = omega/(delta^2+xi)")
    # Self-distillation (LwF) can be combined with all training modes.
    parser.add_argument("--distill", action="store_true",
                        help="Enable self-distillation (LwF-style)")
    parser.add_argument("--distill_alpha", type=float, default=1.0,
                        help="Weight for the self-distillation loss.")
    parser.add_argument("--distill_temperature", type=float, default=5.0,
                        help="Softmax temperature for distillation (paper setting: 5).")
    # Generative Replay
    parser.add_argument("--replay", action="store_true",
                        help="Enable generative replay (Deep Generative Replay)")
    parser.add_argument("--replay_weight", type=float, default=0.75,
                        help="Weight on replay loss (w); current-task CE weight is 1-w")
    parser.add_argument("--replay_n", type=int, default=300,
                        help="Number of replay samples generated per task.")
    parser.add_argument("--replay_temperature", type=float, default=2.0,
                        help="Softmax temperature for replay distillation loss")
    parser.add_argument("--replay_gen_temperature", type=float, default=1.0,
                        help="Sampling temperature for replay text generation")
    parser.add_argument("--replay_max_tokens", type=int, default=None,
                        help="Max new tokens for replay generation. Defaults to "
                             "--max_seq_len so replayed sequences aren't truncated "
                             "shorter than real training sequences; pass a value "
                             "to override.")
    parser.add_argument("--replay_batch_size", type=int, default=32,
                        help="Batch size for replay generation")
    parser.add_argument("--replay_token", action="store_true",
                        help="Use a frozen in-distribution replay token for "
                             "unconditional replay. The token is prepended to "
                             "every training sequence (so the model/generator "
                             "learns <replay> -> Question:...Answer:...) and used "
                             "to seed generation. Its mean-initialized embedding "
                             "remains frozen.")
    # Merge-and-restart LoRA (rank Nr after N tasks instead of capped at r)
    parser.add_argument("--merge_lora_per_task", action="store_true",
                        help="At each task boundary, merge the current LoRA into "
                             "the base model and re-initialise a fresh LoRA "
                             "adapter for the next task. Effective update rank "
                             "grows up to Nr after N tasks instead of staying "
                             "capped at r. Each checkpoint is saved as a full "
                             "merged HF model (evaluate.py auto-detects the format).")
    parser.add_argument("--warmup_frac", type=float, default=0.05,
                        help="Linear LR warmup fraction applied at the start "
                             "of each task (ReLoRA-style jagged schedule). "
                             "E.g. 0.05 = first 5%% of "
                             "optimizer steps ramp from lr/W up to args.lr "
                             "(W = warmup steps; the first step is lr/W, not "
                             "0 — a 0-LR AdamW step would be a wasted no-op). "
                             "0.0 disables warmup. Applies to every task loop, "
                             "not just --merge_lora_per_task.")
    parser.add_argument("--merge_lora_osrm", action="store_true",
                        help="Data-driven orthogonal init from OSRM (Zhang & "
                             "Zhou, ACL 2025), adapted to sequential CL. After "
                             "each task t, capture the mean input feature into "
                             "each LoRA module on a sample of task t's data; "
                             "at start of task t+1, init the fresh A's to the "
                             "smallest eigenvectors of the past-task feature "
                             "covariance (so A·x ≈ 0 for past-task inputs). "
                             "Memory-efficient (one averaged vector per task "
                             "per module, ~25MB total). Requires "
                             "--merge_lora_per_task.")
    parser.add_argument("--olora", action="store_true",
                        help="O-LoRA mode (Wang et al., 2023): retain frozen "
                             "task adapters, train one new adapter per task, "
                             "and apply the official implementation's L1 "
                             "subspace-overlap penalty. Mutually exclusive "
                             "with --merge_lora_per_task.")
    parser.add_argument("--olora_lambda", type=float, default=0.5,
                        help="Weight λ₁ on the O-LoRA orthogonality loss. "
                             "Default 0.5 — matches https://github.com/cmnfriend/O-LoRA "
                             "src/run_uie_lora.py:240 (`lamda_1` default).")
    parser.add_argument("--olora_l2_lambda", type=float, default=0.0,
                        help="Weight λ₂ on the L2 norm of `loranew_*` "
                             "parameters (matches their `lamda_2`). "
                             "The paper setting is 0.")

    # Boxed-answer rendering used by all paper experiments.
    parser.add_argument("--boxed_answers", action="store_true",
                        help="Render every training answer as "
                             "'Answer: \\boxed{<gold>}' (load-time transform of "
                             "train_texts.json rows). Evaluate the resulting "
                             "checkpoints with evaluate.py --boxed_answers "
                             "(first-\\boxed{} exact match). Replay and KD "
                             "inherit the format automatically.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Auto-resume an interrupted run from the last "
                             "completed task using <output_dir>/resume/ "
                             "(method state + RNG, written after every task "
                             "and deleted on completion; see core/resume.py). "
                             "--no-resume clears that state and restarts from "
                             "task 0.")
    # TSH promotion continuation: resume a shorter prior run from
    # an EXTERNAL state dir and extend the horizon (rung b1 -> b2). Unlike
    # --resume (same dir, same n_tasks after a kill), this seeds tasks 0..b1-1
    # from another directory's sidecar + after_task_* checkpoints and trains
    # b1..n_tasks-1. n_tasks is EXCLUDED from the trajectory signature here: the
    # no training hyperparameter depends on the horizon cap, so the first b1
    # tasks are identical.
    parser.add_argument("--resume_from", type=str, default=None,
                        help="External <dir> holding resume/state.pt + "
                             "after_task_* from a shorter prior run; continue it "
                             "at a longer --n_tasks (TSH prefix-resume).")
    parser.add_argument("--start_task", type=int, default=None,
                        help="Assert the resumed start task (must match the "
                             "external state's next_task); guards a wrong "
                             "--resume_from.")
    parser.add_argument("--keep_resume", action="store_true",
                        help="Do NOT delete resume/ on successful completion, so "
                             "this finished run can serve as a --resume_from "
                             "continuation source for a longer rung (TSH).")
    parser.add_argument("--no_mem_log", action="store_true",
                        help="Disable the GPU/RAM usage logger (default: log "
                             "to <output_dir>/mem_log.jsonl; purely "
                             "observational — does not affect training)")
    args = parser.parse_args()

    # Replay generation length tracks --max_seq_len unless explicitly overridden,
    # so replayed sequences aren't capped shorter than real training sequences.
    if args.replay_max_tokens is None:
        args.replay_max_tokens = args.max_seq_len

    # Per-seed output layout is intrinsic to train.py: every run's checkpoints
    # (and train_timing.json) land under <output_dir>/seed_<seed>/ so multiple
    # seeds aggregate cleanly, regardless of how the run was launched.
    args.output_dir = seed_dir(args.output_dir, args.seed)

    if args.olora and args.merge_lora_per_task:
        raise ValueError("--olora and --merge_lora_per_task are mutually exclusive")
    if args.merge_lora_osrm and not args.merge_lora_per_task:
        raise ValueError("--merge_lora_osrm requires --merge_lora_per_task")
    if args.replay and not args.replay_token:
        raise ValueError("The paper's replay method requires --replay_token")
    if args.replay_token and not args.replay:
        raise ValueError("--replay_token is only used with --replay")
    if args.online_EWC and args.olora:
        raise ValueError("The paper evaluates online EWC and O-LoRA separately")
    if args.online_ewc_gamma < 0.0 or args.online_ewc_gamma > 1.0:
        raise ValueError("--online_ewc_gamma must be in [0, 1]")
    # Seed all sources of randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open(os.path.join(args.data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    n_tasks = manifest["n_tasks"]
    if args.n_tasks and args.n_tasks > 0:
        n_tasks = min(n_tasks, args.n_tasks)
        print(f"  Using first {n_tasks} of {manifest['n_tasks']} tasks (--n_tasks)")

    # Task-stream ordering: perm[i] = on-disk task read at stream position i.
    # Built over the FULL horizon (manifest['n_tasks']) so perm[:n_tasks] is a
    # prefix-stable slice across rungs (TSH prefix-resume relies on this).
    # None -> identity = canonical order (unchanged). Checkpoints / accuracy
    # matrix / replay caches stay keyed by STREAM position; only DISK reads map
    # through perm.
    task_perm = task_order(args.task_order_seed, manifest["n_tasks"])
    if args.task_order_seed is not None:
        print(f"  Task order: dev-shuffled (seed {args.task_order_seed}) — "
              f"stream->disk {task_perm[:min(n_tasks, 8)]}"
              f"{' ...' if n_tasks > 8 else ''}")

    use_ewc = args.online_EWC

    os.makedirs(args.output_dir, exist_ok=True)
    _save_train_config(args.output_dir, args)   # record all training HPs (evaluate.py -> hyperparameters.json)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mid-run resume at task boundaries.
    resume_sig = resume_io.build_signature(args, n_tasks)
    resume_state = None
    # resume_src = where the prior state + after_task_* live. For auto-resume
    # (kill recovery) that is output_dir; for TSH promotion-variant
    # continuation it is the shorter prior rung's --resume_from dir.
    # args.output_dir is already seed-nested (seed_dir, above); apply the same to
    # an external --resume_from so callers pass the run-level dir, not seed_<seed>/.
    resume_src = seed_dir(args.resume_from, args.seed) if args.resume_from else args.output_dir
    if args.resume_from:
        # A promoted TSH run is a legitimate shorter prefix, so its horizon is
        # intentionally excluded from the compatibility check.
        resume_state = resume_io.probe(resume_src, resume_sig,
                                       ignore_sig_keys=("__n_tasks_effective", "n_tasks"))
        if resume_state is None:
            raise RuntimeError(f"--resume_from {resume_src} has no resume/state.pt.")
    elif args.resume:
        resume_state = resume_io.probe(args.output_dir, resume_sig)
    else:
        # This run starts from task 0; clear stale state from an earlier run.
        resume_io.clear(args.output_dir)
    write_resume = args.resume
    start_task = 0
    if resume_state is not None:
        start_task = int(resume_state["next_task"])
        if args.start_task is not None and args.start_task != start_task:
            raise RuntimeError(
                f"--start_task {args.start_task} != external state's next_task "
                f"{start_task} — wrong --resume_from?")
        print(f"  Resume: state for tasks 0..{start_task - 1} found in "
              f"{resume_io.state_path(resume_src)} — continuing from "
              f"task {start_task}/{n_tasks}"
              f"{' (continuation)' if args.resume_from else ''}.")

    if not args.no_mem_log:
        from core.mem_log import init_mem_log
        init_mem_log(os.path.join(args.output_dir, "mem_log.jsonl"),
                     meta={"script": "train.py", "argv": sys.argv[1:],
                           "model": args.model, "data_dir": args.data_dir,
                           "n_tasks": args.n_tasks})
        if start_task > 0:
            from core.mem_log import get_mem_log
            get_mem_log().set_task_base(start_task)

    flags = []
    if args.olora:
        l2 = (f", λ2={args.olora_l2_lambda}"
              if args.olora_l2_lambda > 0 else "")
        flags.append(f"O-LoRA(λ1={args.olora_lambda}{l2})")
    if args.merge_lora_per_task:
        wu = (f", wu={args.warmup_frac:.2f}"
              if args.warmup_frac > 0 else "")
        osrm = ", osrm" if args.merge_lora_osrm else ""
        flags.append(f"MergeLoRA(r={args.lora_r}/task{wu}{osrm})")
    elif args.warmup_frac > 0:
        # Warmup applies to every task loop, not just merge restarts.
        flags.append(f"Warmup(frac={args.warmup_frac:.2f})")
    if args.online_EWC:
        norm = ", normF" if args.online_ewc_normalize_fisher else ""
        flags.append(
            f"OnlineEWC(λ={args.ewc_lambda}, γ={args.online_ewc_gamma}{norm})")
    if args.SI:
        flags.append(f"SI(c={args.si_lambda}, xi={args.si_xi}, clamp_neg)")
    if args.distill:
        flags.append(
            f"Distill(forward_kl, α={args.distill_alpha}, T={args.distill_temperature})")
    if args.replay:
        flags.append(
            f"Replay(w={args.replay_weight}, T={args.replay_temperature}, "
            f"gen_T={args.replay_gen_temperature}, loss=kl, mode=unconditional)"
        )
    print(f"Loading {args.model} ...")
    print(f"Regularization: {', '.join(flags) if flags else 'None'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # The merge family folds each task into the base weights, so on resume the
    # live base at the boundary IS the last full-model checkpoint. All other
    # paths keep the pristine base and restore only adapter/method state.
    model_src = args.model
    merge_backbone = args.merge_lora_per_task
    if resume_state is not None and merge_backbone and start_task > 0:
        model_src = os.path.join(resume_src, f"after_task_{start_task - 1}")
        if not os.path.isdir(model_src):
            raise RuntimeError(
                f"resume: {model_src} is missing — merge-family resume needs "
                "the last full-model checkpoint; if checkpoints were pruned "
                "mid-run, restart with --no-resume.")
        print(f"  Resume: loading merged base from {model_src}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_src, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True, device_map=device)

    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Optional in-distribution replay seed token (frozen, LoRA-only) ──
    replay_token = None
    if args.replay_token:
        add_replay_token(tokenizer, base_model)
        replay_token = REPLAY_TOKEN

    modules_to_save = None

    lora_target = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    print(f"LoRA target_modules: {lora_target}")
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=lora_target,
        modules_to_save=modules_to_save,
        bias="none", task_type="CAUSAL_LM")
    if args.olora:
        # O-LoRA uses custom nn.Linear wrappers rather than PEFT adapters.
        model = base_model
    else:
        model = get_peft_model(base_model, lora_cfg)
        model.print_trainable_parameters()

    # ── Re-seed RNG after ALL model/adapter construction, before any task loop ──
    # Building the solver LoRA adapter draws from the global RNG. Without this
    # reset, RNG state consumed during setup would shift task-0 training (a
    # different data shuffle / dropout) and perturb the task-0 diagonal, even
    # though task 0 has no replay and should be identical across variants.
    # Re-seeding here makes the no-replay baseline byte-identical regardless of
    # what was built during setup.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Per-task wall-clock training time (one entry per train_one_task* call),
    # persisted to train_timing.json and surfaced in results.json by evaluate.py.
    task_train_secs: List[float] = []
    if resume_state is not None:
        task_train_secs = [float(s) for s in resume_state["task_train_secs"]]

    if args.olora:
        # ─── O-LoRA (Wang et al., EMNLP-Findings 2023) ────────────────
        # Faithful re-implementation of https://github.com/cmnfriend/O-LoRA
        # using a custom OLoRALinear wrapper that mirrors their forked
        # PEFT's two-pair LoRA structure (frozen accumulated + trainable
        # current). See olora_impl.py for the algorithm cross-references.
        # The upstream get_peft_model was skipped earlier (when args.olora),
        # so `model` is the bare HF transformer with its original Linears.
        from core.olora_impl import (
            wrap_linears_with_olora,
            fold_all_current_into_prior,
            save_olora_loranew,
        )

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        # Freeze all base params; OLoRALinear will manage trainable loranew_*.
        for p in model.parameters():
            p.requires_grad = False
        wrapped_names = wrap_linears_with_olora(
            model, lora_target, args.lora_r, args.lora_alpha,
            args.lora_dropout)
        print(f"  O-LoRA: wrapped {len(wrapped_names)} target Linears")
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  trainable params: {n_train:,} || all params: {n_total:,} "
              f"|| trainable%: {100*n_train/n_total:.4f}")

        if resume_state is not None and start_task > 0:
            # Accumulated frozen buffers from the on-disk per-task factors
            # (concat, NOT fold — fold's loranew re-init would consume RNG);
            # live loranew (post-fold reset values) from the trainable payload.
            n_rebuilt = resume_io.olora_rebuild_accumulated(
                model, resume_src, start_task)
            resume_io.load_trainable(model, resume_state["trainable"])
            resume_io.restore_rng(resume_state)
            print(f"  Resume: rebuilt accumulated O-LoRA rank "
                  f"{start_task * args.lora_r} on {n_rebuilt} layers; "
                  f"continuing at task {start_task}")
        si_state_olora = (SynapticIntelligence(
            xi=args.si_xi,
            clamp_negative=True,
        ) if args.SI else None)
        if resume_state is not None and si_state_olora is not None:
            resume_io.si_restore(si_state_olora, resume_state["si"], device)

        for t in range(start_task, n_tasks):
            print(f"\n{'='*50}")
            print(f"  O-LoRA Training on Task {t} "
                  f"(λ₁={args.olora_lambda}, λ₂={args.olora_l2_lambda})")
            print(f"{'='*50}")

            with open(os.path.join(args.data_dir, f"task_{task_perm[t]}", "train_texts.json")) as f:
                texts = json.load(f)
            if args.boxed_answers:
                texts = box_answer_texts(texts)
            # Prepend the in-distribution replay seed token to every training
            # sequence, matching the standard and merged-LoRA paths. Without
            # this, O-LoRA trains on bare "Question: ..." prompts
            # but evaluate.py prepends <|replay_token|> to every eval prompt, so the
            # just-learned-task diagonal collapses to ~0 (train/eval prompt
            # mismatch; the prefix diverts recall toward replayed/old content).
            if args.replay_token:
                texts = [REPLAY_TOKEN + tx for tx in texts]

            # The frozen accumulated factors reproduce the previous-task model
            # exactly, so snapshot them before optimizing the new factors.
            distiller = None
            if args.distill and t > 0:
                from core.self_distillation import FrozenStateTeacher
                print("  Snapshotting previous-task self-distillation teacher ...")
                distiller = FrozenStateTeacher(
                    model, device, temperature=args.distill_temperature)

            # ── O-LoRA + generative replay (unconditional [S]) ──
            # loranew_B is zero at task start, so the live model == prev task:
            # OLoRAGenerativeReplay snapshots it WITHOUT a PEFT adapter and its
            # in-loop KL teacher zeroes loranew_B on demand.
            replay = None
            replay_texts = None
            if args.replay and t > 0:
                print("  O-LoRA + Replay: snapshotting generator ...")
                replay = OLoRAGenerativeReplay(
                    model, device, temperature=args.replay_temperature,
                    replay_token=replay_token)
                n_replay = args.replay_n
                replay_texts = replay.generate_replay_texts(
                    model, tokenizer, n_replay,
                    max_new_tokens=args.replay_max_tokens,
                    gen_temperature=args.replay_gen_temperature,
                    batch_size=args.replay_batch_size,
                )
                print(f"  O-LoRA + Replay: generated {n_replay} samples")

            # SI snapshots the reference params at task start (no-op unless --SI).
            if si_state_olora is not None:
                si_state_olora.begin_task(model)
            _t0 = time.perf_counter()
            train_one_task_olora(
                model, tokenizer, texts, device, args,
                orth_lambda=args.olora_lambda,
                l2_lambda=args.olora_l2_lambda,
                distiller=distiller,
                replay=replay,
                replay_texts=replay_texts,
                si_state=si_state_olora,
            )
            task_train_secs.append(time.perf_counter() - _t0)
            if replay is not None:
                replay.cleanup()
            if distiller is not None:
                distiller.cleanup()

            # Consolidate SI before the end-of-task fold resets loranew.
            if si_state_olora is not None:
                si_state_olora.consolidate(model)
                print(f"  SI: consolidated {len(si_state_olora.importance)} "
                      f"parameter tensors")
            ckpt_dir = os.path.join(args.output_dir, f"after_task_{t}")
            os.makedirs(ckpt_dir, exist_ok=True)
            # Save only the current task's rank-r loranew
            # factors (~a few hundred MB total over a run). The after-task-t
            # model is reconstructed at eval by summing tasks 0..t
            # (evaluate.py is_olora_factors -> olora_eval_merge). MUST be saved
            # BEFORE the fold, which resets loranew.
            save_olora_loranew(model, ckpt_dir)

            # End-of-task fold: concat current task's loranew into
            # accumulated lora (rank grows by r), reset loranew.
            n_folded = fold_all_current_into_prior(model)
            print(f"  O-LoRA fold: {n_folded} layers; cumulative rank now "
                  f"= (t+1)*r = {(t+1)*args.lora_r}")

            print(f"  O-LoRA checkpoint saved: {ckpt_dir}")

            if write_resume and (t < n_tasks - 1 or args.keep_resume):
                extra = ({"si": resume_io.si_payload(si_state_olora)}
                         if si_state_olora is not None else None)
                resume_io.save_state(
                    args.output_dir, resume_sig, t + 1, task_train_secs,
                    resume_io.collect_trainable(model), extra=extra)

            drop_page_cache(args.output_dir)   # bound page-cache of ckpt writes

        print("\nO-LoRA training complete.")
        _save_train_timing(args.output_dir, n_tasks, task_train_secs)
        if write_resume and not args.keep_resume:
            resume_io.clear(args.output_dir)
        return

    ewc_terms = (OnlineEWC(
        gamma=args.online_ewc_gamma,
        normalize_fisher=args.online_ewc_normalize_fisher,
    ) if args.online_EWC else [])
    si_state = (SynapticIntelligence(
        xi=args.si_xi,
        clamp_negative=True,
    ) if args.SI else None)
    past_features_list: List[dict] = []   # for --merge_lora_osrm (data-driven)

    if resume_state is not None and start_task > 0:
        # For the merge family the base was already reloaded from
        # after_task_<start-1>, so the trainable payload here is the fresh
        # task-<start> adapter (incl. its ortho/OSRM init); for the standard
        # path it is the live adapter as trained through task <start-1>.
        resume_io.load_trainable(model, resume_state["trainable"])
        if use_ewc:
            ewc_terms = resume_io.online_ewc_from_payload(
                resume_state["online_ewc"])
        if si_state is not None:
            resume_io.si_restore(si_state, resume_state["si"], device)
        if args.merge_lora_per_task and args.merge_lora_osrm:
            past_features_list = [
                resume_io.load_task_aux(resume_src, s)["osrm_feat"]
                for s in range(start_task)]
        resume_io.restore_rng(resume_state)

    for t in range(start_task, n_tasks):
        print(f"\n{'='*50}")
        print(f"  Training on Task {t}")
        print(f"{'='*50}")

        with open(os.path.join(args.data_dir, f"task_{task_perm[t]}", "train_texts.json")) as f:
            texts = json.load(f)
        if args.boxed_answers:
            texts = box_answer_texts(texts)

        # Prepend the in-distribution replay seed token to every training
        # sequence so the solver (and any generator trained on these texts)
        # learns the <seed> -> "Question: ... Answer: ..." transition used to
        # seed unconditional replay generation.
        if args.replay_token:
            texts = [REPLAY_TOKEN + tx for tx in texts]

        # ── Snapshot LwF teacher (for any training mode, if t > 0) ──
        distiller = None
        if args.distill and t > 0:
            print(f"  Snapshotting teacher for self-distillation ...")
            distiller = SelfDistillation(model, device,
                                         temperature=args.distill_temperature)

        # ── Generative Replay ──
        replay = None
        replay_texts = None

        if args.replay and t > 0:
            print(f"  Snapshotting teacher for generative replay ...")
            replay = GenerativeReplay(
                model, device, temperature=args.replay_temperature,
                replay_token=replay_token)

            n_replay = args.replay_n

            print(f"  Generating {n_replay} replay samples (unconditional) ...")

            replay_texts = replay.generate_replay_texts(
                model, tokenizer, n_replay,
                max_new_tokens=args.replay_max_tokens,
                gen_temperature=args.replay_gen_temperature,
                batch_size=args.replay_batch_size,
            )

            rpl_log_path = os.path.join(
                args.output_dir, f"task_{t}_replay.json")
            with open(rpl_log_path, "w") as f:
                json.dump(replay_texts, f, indent=2)
            print(f"  Replay data logged to {rpl_log_path}")

        if si_state is not None:
            si_state.begin_task(model)
        _t0 = time.perf_counter()
        if replay is not None:
            train_one_task_replay(
                model, tokenizer, texts, replay_texts, device, args,
                replay=replay,
                ewc_terms=ewc_terms if use_ewc and ewc_terms else None,
                distiller=distiller,
                si_state=si_state,
            )
        else:
            train_one_task(model, tokenizer, texts, device, args,
                           ewc_terms=ewc_terms if use_ewc and ewc_terms else None,
                           distiller=distiller,
                           si_state=si_state)
        task_train_secs.append(time.perf_counter() - _t0)

        if replay is not None:
            replay.cleanup()

        if distiller is not None:
            distiller.cleanup()

        if si_state is not None:
            si_state.consolidate(model)
            print(f"  SI: consolidated {len(si_state.importance)} parameter tensors")

        if use_ewc:
            print(f"  OnlineEWC: updating running Fisher for task {t} ...")
            ewc_terms.consolidate(
                model, tokenizer, texts, device,
                max_seq_len=args.max_seq_len,
                n_samples=args.ewc_n_samples)
            print(f"  OnlineEWC: consolidated {len(ewc_terms.fisher)} "
                  f"parameter tensors from {ewc_terms.n_tasks} task(s)")

        ckpt_dir = os.path.join(args.output_dir, f"after_task_{t}")
        if args.merge_lora_per_task:
            # Snapshot per-task state BEFORE merge destroys the adapter.
            if args.merge_lora_osrm:
                print(f"  Collecting OSRM input features for task {t} ...")
                past_features_list.append(
                    collect_lora_input_features(
                        model, tokenizer, texts, device,
                        n_samples=min(64, len(texts)),
                        max_seq_len=args.max_seq_len,
                        batch_size=max(1, args.batch_size)))

            print(f"  Merging LoRA into base for task {t} ...")
            merged = model.merge_and_unload()
            merged.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  Merged checkpoint saved: {ckpt_dir}")

            if t < n_tasks - 1 or args.keep_resume:
                # A completed non-final TSH rung remains continuation-ready.
                if hasattr(merged, "enable_input_require_grads"):
                    merged.enable_input_require_grads()
                model = get_peft_model(merged, lora_cfg)
                if args.merge_lora_osrm:
                    n = osrm_orthogonal_init_A(model, past_features_list)
                    print(f"  OSRM-initialised {n} LoRA A matrices from "
                          f"{len(past_features_list)} past-task features")
                model.print_trainable_parameters()
        else:
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  Checkpoint saved: {ckpt_dir}")

        if write_resume and (t < n_tasks - 1 or args.keep_resume):
            # Immutable per-task contributions first, then the running state
            # (whose next_task pointer is what commits the task as done).
            aux = {}
            if args.merge_lora_per_task and args.merge_lora_osrm:
                aux["osrm_feat"] = past_features_list[-1]
            resume_io.save_task_aux(args.output_dir, t, aux)
            extra = {}
            if args.online_EWC:
                extra["online_ewc"] = resume_io.online_ewc_payload(ewc_terms)
            if si_state is not None:
                extra["si"] = resume_io.si_payload(si_state)
            resume_io.save_state(args.output_dir, resume_sig, t + 1,
                                 task_train_secs,
                                 resume_io.collect_trainable(model),
                                 extra=extra)

        # Bound page-cache use after this task's checkpoint/resume writes.
        drop_page_cache(args.output_dir)

    _save_train_timing(args.output_dir, n_tasks, task_train_secs)
    if write_resume and not args.keep_resume:
        resume_io.clear(args.output_dir)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
