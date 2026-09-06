"""Mathematical reasoning evaluation for ComposeCL.

Greedy few-shot chain-of-thought generation, then answer extraction
(`\\boxed{}` -> `#### N` / "answer is X" -> last number) and sympy-based
equivalence grading via the vendored math grader (evals/math_grader.py).

Loads a base model and optional PEFT adapter, with room for chain-of-thought
generation (max_new_tokens=512) and math grading.

Usage:
  eval_math.py --base_model Qwen/Qwen3-0.6B-Base \
      [--adapter <after_task_9_dir>] \
      --eval_jsonl data/general_eval/math/gsm8k/eval.jsonl \
      --output results/math/<model>__<ckpt>__gsm8k.json \
      --batch_size 32
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))   # repo root (this file lives in evals/)
from evals.math_grader import grade_answer  # noqa: E402
from evals.math_normalize import remove_boxed  # noqa: E402


# 4-shot CoT exemplars (MATH-style, \boxed final answers). Same prefix for all
# three datasets so the pre-SFT vs after-SFT comparison is apples-to-apples.
FEWSHOT = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as "
     "many clips in May. How many clips did she sell altogether in April and "
     "May?",
     "In April she sold 48 clips. In May she sold half as many, 48 / 2 = 24. "
     "Altogether 48 + 24 = 72. The final answer is \\boxed{72}."),
    ("What is the value of $3^2 + 4^2$?",
     "We compute 3^2 = 9 and 4^2 = 16. Their sum is 9 + 16 = 25. "
     "The final answer is \\boxed{25}."),
    ("If $2x + 5 = 13$, what is the value of $x$?",
     "Subtract 5 from both sides: 2x = 8. Divide by 2: x = 4. "
     "The final answer is \\boxed{4}."),
    ("A rectangle has length 8 and width 3. What is its area?",
     "Area = length times width = 8 * 3 = 24. "
     "The final answer is \\boxed{24}."),
]


def build_prefix() -> str:
    parts = []
    for q, sol in FEWSHOT:
        parts.append(f"Problem:\n{q}\n\nSolution:\n{sol}")
    return "\n\n".join(parts) + "\n\n"


PREFIX = build_prefix()
STOP = "\nProblem:"


def make_prompt(problem: str) -> str:
    return f"{PREFIX}Problem:\n{problem}\n\nSolution:\n"


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    num_left = 0
    seen_brace = False
    right = None
    while i < len(string):
        if string[i] == "{":
            num_left += 1
            seen_brace = True
        elif string[i] == "}":
            num_left -= 1
            if seen_brace and num_left == 0:
                right = i
                break
        i += 1
    if right is None:
        return None
    return string[idx:right + 1]


_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_answer(gen: str) -> str:
    """Extract the final answer from a generated solution."""
    # Cut at the start of a hallucinated next problem.
    if STOP in gen:
        gen = gen.split(STOP)[0]
    # 1. last \boxed{}
    boxed = last_boxed_only_string(gen)
    if boxed is not None:
        inner = None
        try:
            inner = remove_boxed(boxed)
        except Exception:
            inner = None
        if not (inner and inner.strip()):
            m = re.match(r"\\boxed\{(.*)\}$", boxed, re.DOTALL)
            inner = m.group(1) if m else None
        if inner and inner.strip():
            return inner.strip()
    # 2. GSM8K-style #### N
    m = re.search(r"####\s*(.+)", gen)
    if m:
        return m.group(1).strip().rstrip(".")
    # 3. "(final )answer is X"
    m = re.search(r"answer is[:\s]*\$?(.+?)(?:\.|\n|$)", gen, re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip().rstrip(".$")
    # 4. last number-ish token
    nums = _NUM_RE.findall(gen)
    if nums:
        return nums[-1].replace(",", "").replace("$", "").strip()
    return gen.strip()[-64:]


def make_mv_grader():
    """HuggingFace Math-Verify grader: parse the FULL generation (its own robust
    \\boxed/last-answer extraction) and the gold, then check math equivalence.
    Gold is wrapped in $...$ so LaTeX extraction fires for latex golds (e.g.
    \\frac{3}{2}); falls back to bare parse. Returns a grade(generation, gold)->bool.
    Validated on tricky cases (frac==decimal, sets, surds) in test/smoke."""
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    cfg = [LatexExtractionConfig(), ExprExtractionConfig()]

    def grade(generation: str, gold: str) -> bool:
        try:
            g = parse(f"${gold}$", cfg) or parse(gold, cfg)
            p = parse(generation, cfg)
            return bool(verify(g, p))
        except Exception:
            return False
    return grade


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p) if l.strip()]


def _pct(xs, q):
    """q-th percentile (nearest-rank) of a list of ints; 0 if empty."""
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--olora_factors", default=None,
                    help="O-LoRA disk-light checkpoint ROOT (the seed_42 dir "
                         "holding after_task_*/olora_loranew.safetensors). "
                         "Reconstructs the model after --olora_upto by summing "
                         "the per-task factors into --base_model in place, "
                         "mirroring evaluate.py's disk-light eval. Mutually "
                         "exclusive with --adapter.")
    ap.add_argument("--olora_upto", type=int, default=None,
                    help="Final task index for --olora_factors (the model after "
                         "this many tasks; required with --olora_factors).")
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only eval the first N problems (for sanity).")
    ap.add_argument("--grader", choices=["sympy", "math_verify"], default="sympy",
                    help="Answer-equivalence checker. 'sympy' (default) = the "
                         "in-repo extract_answer + math_grader.grade_answer, kept "
                         "as default so existing numbers stay reproducible. "
                         "'math_verify' = HuggingFace Math-Verify (parses the full "
                         "generation, robust LaTeX/sympy equivalence: frac==decimal, "
                         "sets, surds, etc.). See evals/regrade_math.py to "
                         "re-grade saved runs offline.")
    args = ap.parse_args()
    mv_grade = make_mv_grader() if args.grader == "math_verify" else None

    if os.path.exists(args.output):
        print(f"  skip: {args.output} exists")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True, device_map=device)
    tag = "base"
    if args.olora_factors:
        if args.adapter:
            raise SystemExit("--olora_factors and --adapter are mutually exclusive")
        if args.olora_upto is None:
            raise SystemExit("--olora_factors requires --olora_upto")
        from core.olora_impl import olora_eval_merge
        nlay = olora_eval_merge(model, args.olora_factors, args.olora_upto)
        tag = "olora_factors"
        print(f"  O-LoRA: merged tasks 0..{args.olora_upto} into base "
              f"({nlay} layers)")
    elif args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        tag = "adapter"
    model.eval()

    rows = load_jsonl(Path(args.eval_jsonl))
    if args.limit > 0:
        rows = rows[:args.limit]
    # gsm8k/math/aime carry "problem"; mgsm (general_eval) carries "question".
    prompts = [make_prompt(r.get("problem") or r.get("question", "")) for r in rows]

    records = []
    n_correct = 0
    # --- truncation logging: count generations that hit the max_new_tokens cap
    # (i.e. cut off mid-solution, before the final \boxed{}) so we can tell
    # whether the generation budget is large enough for THIS eval set. ---
    eos_id = tok.eos_token_id
    gen_len_all = []
    n_trunc = 0
    n_trunc_no_box = 0
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        bp = prompts[start:start + args.batch_size]
        br = rows[start:start + args.batch_size]
        enc = tok(bp, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
                # base models have no EOS supervision and ramble a NEW "Problem:"
                # after answering; stop there so the solution is just this problem.
                stop_strings=[STOP], tokenizer=tok)
        gen_ids = out[:, enc["input_ids"].shape[1]:]
        gens = tok.batch_decode(gen_ids, skip_special_tokens=True)
        # Per-sequence generated length. Greedy decoding only ever emits the
        # pad/eos id as a TRAILING run (it halts on eos/stop_string), so counting
        # non-pad tokens recovers the true generated length per row. A row whose
        # length reaches the cap stopped for no other reason => truncated.
        gen_lens = (gen_ids != tok.pad_token_id).sum(dim=1).tolist()
        eos_present = ((gen_ids == eos_id).any(dim=1).tolist()
                       if eos_id is not None else [False] * len(gens))
        for i, (r, g) in enumerate(zip(br, gens)):
            # Grade only the solution to THIS problem: cut at the first rambled
            # "\nProblem:" (belt-and-suspenders with stop_strings) so the grader
            # never picks the \boxed{} of a follow-up problem the model invented.
            sol = g.split(STOP)[0]
            glen = gen_lens[i]
            truncated = glen >= args.max_new_tokens     # hit the cap, not a natural stop
            has_box = last_boxed_only_string(sol) is not None
            if truncated:
                stop_reason = "length_cap"              # ran out of budget mid-solution
            elif STOP in g:
                stop_reason = "stop_string"             # rambled into a new "Problem:"
            elif eos_present[i]:
                stop_reason = "eos"                      # model ended on its own
            else:
                stop_reason = "other"
            gen_len_all.append(glen)
            n_trunc += int(truncated)
            n_trunc_no_box += int(truncated and not has_box)
            pred = extract_answer(sol)          # readable extracted answer (logged)
            gold = str(r["answer"])
            if mv_grade is not None:
                ok = mv_grade(sol, gold)        # Math-Verify parses the solution span
            else:
                try:
                    ok = bool(grade_answer(pred, gold))
                except Exception:
                    ok = False
            n_correct += int(ok)
            problem = r.get("problem") or r.get("question", "")
            records.append({
                "qid": r["qid"], "dataset": r["dataset"],
                # formatted item question (no chat template); the constant 4-shot
                # prefix is stored once in summary["prompt_prefix"], so the full
                # model input = prompt_prefix + question.
                "question": f"Problem:\n{problem}\n\nSolution:\n",
                # FULL generation (no truncation) so the run is always re-gradeable
                # offline (regrade_math.py) without re-running the model.
                "generation": g,
                "parsed_answer": pred,
                "gold": gold,
                "correct": ok,
                "n_gen_tokens": int(glen),
                "truncated": bool(truncated),
                "stop_reason": stop_reason,
            })
        done = start + len(br)
        print(f"    {done}/{len(rows)}  acc={n_correct/done:.4f}", flush=True)

    summary = {
        "n": len(rows),
        "accuracy": n_correct / max(len(rows), 1),
        "n_correct": n_correct,
        "grader": args.grader,
        "prompt_prefix": PREFIX,     # constant 4-shot CoT prefix (full input = prefix+question)
        "max_new_tokens": args.max_new_tokens,
        # truncation diagnostics: is the generation budget big enough here?
        "n_truncated": n_trunc,
        "trunc_frac": n_trunc / max(len(rows), 1),
        "n_trunc_no_answer": n_trunc_no_box,   # truncated AND emitted no \boxed{}
        "gen_tokens": {
            "max": max(gen_len_all) if gen_len_all else 0,
            "mean": round(sum(gen_len_all) / max(len(gen_len_all), 1), 1),
            "p50": _pct(gen_len_all, 0.50),
            "p95": _pct(gen_len_all, 0.95),
            "p99": _pct(gen_len_all, 0.99),
        },
        "elapsed_s": time.time() - t0,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "tag": tag,
        "eval_jsonl": args.eval_jsonl,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump({"summary": summary, "records": records},
              open(args.output, "w"), ensure_ascii=False)
    print(f"  DONE  n={summary['n']}  acc={summary['accuracy']:.4f}  "
          f"-> {args.output}")
    gt = summary["gen_tokens"]
    print(f"  GEN TOKENS  p50={gt['p50']} p95={gt['p95']} p99={gt['p99']} "
          f"max={gt['max']}  (cap={args.max_new_tokens})")
    if n_trunc:
        print(f"  ⚠ TRUNCATION  {n_trunc}/{summary['n']} "
              f"({100 * summary['trunc_frac']:.1f}%) hit the {args.max_new_tokens}-tok "
              f"cap; {n_trunc_no_box} produced NO \\boxed answer (accuracy lost to "
              f"truncation). Consider raising --max_new_tokens.")
    else:
        print(f"  ✓ no truncation: 0/{summary['n']} hit the "
              f"{args.max_new_tokens}-tok cap.")


if __name__ == "__main__":
    main()
