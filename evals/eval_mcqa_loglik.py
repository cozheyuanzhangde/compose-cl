#!/usr/bin/env python
"""Letter-log-likelihood evaluation for MMLU-Redux.

Each prompt ends in ``Answer:`` and the evaluator scores the next-token
likelihood of every valid answer letter. Argmax is the prediction, so the metric
does not depend on free-form output parsing.

Loads a base HF model OR a full-model checkpoint (--model), + optional PEFT
--adapter. Output schema matches the other evals (full per-item logging).
Usage: eval_mcqa_loglik.py --model <hf-or-ckpt> [--adapter A] --eval_jsonl J --output O
"""
import argparse, json, os, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))   # repo root (for core.olora_impl)
from evals.mcqa_common import LETTERS, PREFIX, format_question, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)        # base HF id OR full-model ckpt dir
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--olora_factors", default=None,
                    help="O-LoRA disk-light checkpoint ROOT (seed_42 dir with "
                         "after_task_*/olora_loranew.safetensors); reconstructs "
                         "the model after --olora_upto into --model in place. "
                         "Mutually exclusive with --adapter.")
    ap.add_argument("--olora_upto", type=int, default=None,
                    help="Final task index for --olora_factors.")
    ap.add_argument("--tokenizer", default=None)     # default: --model
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if os.path.exists(args.output):
        print(f"  skip: {args.output} exists"); return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
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
        model = PeftModel.from_pretrained(model, args.adapter); tag = "adapter"
    model.eval()

    # token id of the FIRST token of " A", " B", ... (the trained "Answer: <L>" form)
    letter_tok = [tok(" " + L, add_special_tokens=False)["input_ids"][0] for L in LETTERS]

    rows = load_jsonl(Path(args.eval_jsonl))
    if args.limit > 0:
        rows = rows[:args.limit]
    prompts = [PREFIX + format_question(r["question"], r["choices"]) for r in rows]

    records, n_correct = [], 0
    t0 = time.time()
    for s in range(0, len(rows), args.batch_size):
        bp, br = prompts[s:s+args.batch_size], rows[s:s+args.batch_size]
        enc = tok(bp, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()    # next-token logits (left-padded)
        logprobs = torch.log_softmax(logits, dim=-1)
        for i, r in enumerate(br):
            nc = len(r["choices"])
            lp = [logprobs[i, letter_tok[k]].item() for k in range(nc)]
            pred_idx = int(max(range(nc), key=lambda k: lp[k]))
            ok = (pred_idx == r["answer_idx"])
            n_correct += int(ok)
            records.append({
                "qid": r["qid"],
                "question": format_question(r["question"], r["choices"]),
                "letter_logprobs": {LETTERS[k]: round(lp[k], 4) for k in range(nc)},
                "parsed_answer": LETTERS[pred_idx],
                "gold": LETTERS[r["answer_idx"]],
                "gold_idx": r["answer_idx"], "pred_idx": pred_idx, "correct": bool(ok),
            })
        print(f"    {s+len(br)}/{len(rows)}  acc={n_correct/(s+len(br)):.4f}", flush=True)

    n = max(1, len(rows))
    summary = {"n": len(rows), "accuracy": n_correct / n, "n_correct": n_correct,
               "decided_frac": 1.0, "method": "loglik_letter",
               "model": args.model, "adapter": args.adapter, "tag": tag,
               "eval_jsonl": args.eval_jsonl, "prompt_prefix": PREFIX,
               "elapsed_s": time.time() - t0}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(args.output, "w"), ensure_ascii=False)
    print(f"  DONE  n={summary['n']}  loglik_acc={summary['accuracy']:.4f} -> {args.output}")


if __name__ == "__main__":
    main()
