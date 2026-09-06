"""
LLM-knowledge QA generator: LLM-synthesized NOVEL knowledge for continual
learning.

The most realistic tier of the memorization suite. A local LLM (Qwen via
`transformers`) invents NEW facts about FICTIONAL entities, written in natural,
real-word language — facts a pretrained model does not already know, so they are
genuinely new knowledge to acquire. Contrast the suite:

    symbol_qa      arbitrary symbol associations          (no semantics)
    templated_qa   templated QA with fabricated fillers    (language form, fake content)
    llm_qa         LLM-synthesized novel knowledge         (real words, invented facts)   <-- here

Why "real words but invented facts"? If you ask an LLM for *real* facts, it emits
things it already knows, so the base model answers them and there is nothing to
learn (base accuracy high → no forgetting signal). The trick is real vocabulary,
fictional specifics: fluent prose about entities/relationships that do not exist.
A base model gets these wrong; after training it should recall them; later tasks
overwrite them → clean catastrophic-forgetting signal. Use `--novelty_filter_model`
to additionally drop any item the base model already answers, guaranteeing novelty.

Train == test (memorization), 100 tasks, 100 QA/task, repeat=1 by default.

Throughput
----------
Generation batches prompts ACROSS tasks into one model.generate call, and the
batch size is auto-sized to fill the GPU (see `--parallel_prompts 0`). A single
task only needs a handful of prompts, so without cross-task batching a big GPU
sits idle; pooling prompts from many tasks keeps it saturated.

Disambiguation (why every test query has exactly one answer)
-----------------------------------------------------------
At test time the model sees only `Question: <q>\nAnswer:` with NO task label, so
the global map question -> answer MUST be single-valued across all tasks. Every
question is anchored on an invented ENTITY that must appear verbatim in the
question; we keep entities (and questions) globally unique via a registry and a
final verification pass, so each question determines exactly one answer.

Output format matches `train.py` / `evaluate.py`:
  task_t/train_texts.json  - List[str], each "Question: <q>\nAnswer: <a>"
  task_t/train_items.json  - List[Dict] with id/query/answer/train_text/...
  task_t/test_items.json   - List[Dict] (identical to train)
  manifest.json            - n_tasks + per-task metadata

Usage (real, on a GPU):
    python data_generation/synthetic_qa/llm_qa.py --n_tasks 100 --items_per_task 100 \
        --backend qwen --generator_model Qwen/Qwen3-4B-Instruct-2507

Offline plumbing smoke test (no model needed):
    python data_generation/synthetic_qa/llm_qa.py --n_tasks 3 --items_per_task 10 --backend template
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Tuple

DEFAULT_SPEC = os.path.join(os.path.dirname(__file__), "llm_qa_topics.json")

SYSTEM_PROMPT = (
    "You generate synthetic benchmark data. You invent fictional, non-existent "
    "facts and output ONLY a valid JSON array — no prose, no markdown fences, "
    "no comments."
)


# --------------------------------------------------------------------------- #
# Spec + name machinery
# --------------------------------------------------------------------------- #

def _load_topics(spec_path: str) -> List[Dict]:
    with open(spec_path) as f:
        spec = json.load(f)
    topics = spec["topics"]
    if not topics:
        raise ValueError(f"No topics found in {spec_path}")
    return topics


# --------------------------------------------------------------------------- #
# Robust JSON-array extraction from a (possibly messy) LLM completion
# --------------------------------------------------------------------------- #

def _extract_json_array(text: str) -> List:
    """Pull the first valid JSON array of objects out of an LLM completion."""
    cleaned = text.strip()
    # Drop any <think>...</think> and markdown fences.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.I | re.S).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates: List[str] = []
    first, last = cleaned.find("["), cleaned.rfind("]")
    if first != -1 and last != -1 and last > first:
        candidates.append(cleaned[first:last + 1])
    candidates.append(cleaned)

    for cand in candidates:
        try:
            payload = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("items", "data", "facts", "questions"):
                if isinstance(payload.get(key), list):
                    return payload[key]
    return []


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _clean_candidate(raw, max_answer_words: int) -> Optional[Dict]:
    """Validate one raw {entity, question, answer} object."""
    if not isinstance(raw, dict):
        return None
    entity = str(raw.get("entity", "")).strip()
    question = str(raw.get("question", "")).strip()
    answer = str(raw.get("answer", "")).strip()
    if not entity or not question or not answer:
        return None
    # The unique entity must be inside the question (it is the disambiguator).
    if _norm(entity) not in _norm(question):
        return None
    if not question.endswith("?"):
        question = question.rstrip(".") + "?"
    if len(answer.split()) > max_answer_words:
        return None
    if "\n" in question or "\n" in answer:
        return None
    return {"entity": entity, "question": question, "answer": answer}


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

def _gen_prompt(topic_name: str, n: int, seed: int) -> str:
    return (
        f"Invent {n} fictional, plausible-sounding but ENTIRELY NON-EXISTENT "
        f"facts about {topic_name}.\n\n"
        "These must be NEW invented knowledge: do NOT use any real people, "
        "places, organizations, works, or events. Invent original proper names. "
        "The facts should read naturally and use ordinary English words, but the "
        "specific entities and facts must not exist in the real world.\n\n"
        f"Return ONLY a JSON array of {n} objects. Each object must have exactly "
        "these keys:\n"
        '  "entity":   a unique invented proper name the question is about\n'
        '  "question": a self-contained question that NAMES the entity verbatim '
        "and has a single correct short answer\n"
        '  "answer":   the short answer (1-6 words): an invented name, place, '
        "number, year, or short phrase\n\n"
        "Rules:\n"
        "- The exact \"entity\" string MUST appear verbatim inside \"question\".\n"
        "- Every entity must be distinct and not a real-world entity.\n"
        "- Vary question forms (who / what / which / when / where / how many).\n"
        "- Keep each answer short and specific.\n"
        "- Output must start with '[' and end with ']'. Use double quotes. No "
        "trailing commas, no comments, no markdown.\n\n"
        f"Diversity seed: {seed}"
    )


# --------------------------------------------------------------------------- #
# Backends. Each exposes generate(requests) -> List[List[Dict]], where a request
# is (topic_name, n_items, seed) and the result is the parsed candidate list for
# that prompt. Cross-task batching lives in the orchestrator, which hands a whole
# batch of requests (drawn from many tasks) to one generate() call.
# --------------------------------------------------------------------------- #

class QwenGenerator:
    """Local Hugging Face causal-LM backend (default: a Qwen instruct model)."""

    def __init__(self, model_name: str, temperature: float, top_p: float,
                 max_new_tokens: int, parallel_prompts: int,
                 mem_frac: float = 0.7, hard_cap: int = 64):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The qwen backend needs torch + transformers. Install them, or "
                "use --backend template for an offline smoke test.") from exc
        self.torch = torch
        self.temperature, self.top_p, self.max_new_tokens = temperature, top_p, max_new_tokens

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"  Loading generator model {model_name} on {device} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
            device_map="auto" if device == "cuda" else None)
        if device != "cuda":
            self.model.to(device)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.eval()
        self.input_device = next(self.model.parameters()).device

        if parallel_prompts and parallel_prompts > 0:
            self.parallel_prompts = parallel_prompts
            print(f"  Generation batch (parallel_prompts) = {self.parallel_prompts} (manual)")
        else:
            self.parallel_prompts = self._auto_batch(mem_frac, hard_cap)
            print(f"  Generation batch (parallel_prompts) = {self.parallel_prompts} "
                  f"(auto-sized to ~{int(mem_frac*100)}% of free VRAM, cap {hard_cap})")

    def _auto_batch(self, mem_frac: float, hard_cap: int, prompt_tokens: int = 512) -> int:
        """Pick the largest prompt batch that fits a fraction of free VRAM.

        Dominant per-sequence cost is the KV cache over (prompt + generated)
        tokens; a 30% pad covers prefill activations, logits and fragmentation.
        """
        if self.input_device.type != "cuda":
            return 8
        cfg = self.model.config
        n_layers = getattr(cfg, "num_hidden_layers", 32)
        n_heads = getattr(cfg, "num_attention_heads", 32)
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
        hidden = getattr(cfg, "hidden_size", n_heads * 128)
        head_dim = getattr(cfg, "head_dim", None) or (hidden // max(n_heads, 1))
        kv_bytes_per_tok = 2 * n_layers * n_kv * head_dim * 2  # k & v, bf16
        per_seq = int(kv_bytes_per_tok * (prompt_tokens + self.max_new_tokens) * 1.3)
        free, _total = self.torch.cuda.mem_get_info()
        n = int(free * mem_frac) // max(per_seq, 1)
        return int(max(4, min(n, hard_cap)))

    def _format(self, prompt: str) -> str:
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        return f"{SYSTEM_PROMPT}\n\n{prompt}\n"

    def generate(self, requests: List[Tuple[str, int, int]]) -> List[List[Dict]]:
        results: List[List[Dict]] = []
        # Chunk to the memory-safe batch size, even if handed more.
        for start in range(0, len(requests), self.parallel_prompts):
            chunk = requests[start:start + self.parallel_prompts]
            prompts = [self._format(_gen_prompt(name, n, seed)) for (name, n, seed) in chunk]
            side = self.tok.padding_side
            self.tok.padding_side = "left"
            try:
                enc = self.tok(prompts, return_tensors="pt", padding=True,
                               truncation=False).to(self.input_device)
            finally:
                self.tok.padding_side = side
            with self.torch.inference_mode():
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=True,
                    temperature=self.temperature, top_p=self.top_p,
                    pad_token_id=self.tok.pad_token_id,
                    eos_token_id=self.tok.eos_token_id)
            width = enc["input_ids"].shape[1]
            for i in range(len(chunk)):
                txt = self.tok.decode(out[i, width:], skip_special_tokens=True)
                results.append(_extract_json_array(txt))
        return results


class TemplateBackend:
    """Offline deterministic fallback for plumbing smoke tests (NOT real knowledge).

    Fabricates unique {entity, question, answer} triples from word pools. The
    text uses real English words and invented proper names, but it is not
    LLM-generated and is not novelty-checked — use only to exercise the pipeline.
    """
    SYL = ["ar", "bel", "cor", "dai", "el", "fen", "gal", "hal", "ion", "jor",
           "kai", "lor", "mir", "nel", "or", "pax", "qua", "rin", "sol", "tor",
           "ul", "ves", "wyn", "xan", "yor", "zen", "ash", "bre", "cle", "dov"]
    KIND = ["League", "Institute", "Expedition", "Codex", "Accord", "Order",
            "Survey", "Company", "Academy", "Guild", "Observatory", "Archive"]
    FRAMES = [
        ("Who founded {e}?", "person"),
        ("In what year was {e} established?", "year"),
        ("Which city is the seat of {e}?", "place"),
        ("How many members did {e} originally have?", "number"),
        ("What is the motto of {e}?", "phrase"),
        ("Who currently leads {e}?", "person"),
    ]
    # Auto-sizing is GPU-only; the template backend just uses a fixed batch.
    parallel_prompts = 16

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def _coin(self, n_syl: int) -> str:
        return "".join(self.rng.choice(self.SYL) for _ in range(n_syl)).capitalize()

    def _answer(self, kind: str) -> str:
        if kind == "year":
            return str(self.rng.randint(1400, 1980))
        if kind == "number":
            return str(self.rng.randint(7, 480))
        if kind == "person":
            return f"{self._coin(2)} {self._coin(self.rng.randint(2, 3))}"
        if kind == "place":
            return f"{self._coin(self.rng.randint(2, 3))}"
        return f"{self._coin(2)} {self._coin(2)}"  # phrase

    def generate(self, requests: List[Tuple[str, int, int]]) -> List[List[Dict]]:
        out: List[List[Dict]] = []
        for (_name, n, _seed) in requests:
            items = []
            for _ in range(n):
                entity = f"the {self._coin(self.rng.randint(2, 3))} {self.rng.choice(self.KIND)}"
                frame, kind = self.rng.choice(self.FRAMES)
                items.append({"entity": entity,
                              "question": frame.format(e=entity),
                              "answer": self._answer(kind)})
            out.append(items)
        return out


# --------------------------------------------------------------------------- #
# Optional novelty filter: keep only items the base model does NOT already know
# --------------------------------------------------------------------------- #

class NoveltyFilter:
    def __init__(self, model_name: str, batch_size: int, max_new_tokens: int = 32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"  Loading novelty-filter model {model_name} on {device} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
            device_map="auto" if device == "cuda" else None)
        if device != "cuda":
            self.model.to(device)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.eval()
        self.input_device = next(self.model.parameters()).device

    def is_novel(self, items: List[Dict]) -> List[bool]:
        """True for items the base model answers WRONG (i.e. genuinely novel)."""
        verdicts: List[bool] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            prompts = [f"Question: {it['question']}\nAnswer:" for it in batch]
            side = self.tok.padding_side
            self.tok.padding_side = "left"
            try:
                enc = self.tok(prompts, return_tensors="pt", padding=True,
                               truncation=True).to(self.input_device)
            finally:
                self.tok.padding_side = side
            with self.torch.inference_mode():
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.pad_token_id)
            width = enc["input_ids"].shape[1]
            for i, it in enumerate(batch):
                gen = self.tok.decode(out[i, width:], skip_special_tokens=True)
                verdicts.append(it["answer"].lower() not in gen.lower())
        return verdicts


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def _build_item(task_id: int, topic: Dict, idx: int, cand: Dict) -> Dict:
    qa = f"Question: {cand['question']}\nAnswer: {cand['answer']}"
    return {
        "id": f"task_{task_id}_item_{idx}",
        "task_id": task_id,
        "topic": topic["id"],
        "category": topic["id"],
        "query": cand["question"],
        "answer": cand["answer"],
        "train_text": qa,
        "entity": cand["entity"],
    }


def make_train_texts(items: List[Dict], n_repeat: int, rng: random.Random) -> List[str]:
    texts: List[str] = []
    for item in items:
        qa = f"Question: {item['query']}\nAnswer: {item['answer']}"
        texts.extend([qa] * n_repeat)
    rng.shuffle(texts)
    return texts


def verify_no_global_ambiguity(all_items: List[List[Dict]]) -> None:
    q2a: Dict[str, str] = {}
    n = 0
    for task_items in all_items:
        for item in task_items:
            q, a = item["query"], item["answer"]
            assert q and a, f"empty query/answer in {item['id']}"
            if q in q2a and q2a[q] != a:
                raise AssertionError(
                    f"AMBIGUOUS query {q!r}: maps to {q2a[q]!r} and {a!r}")
            q2a[q] = a
            n += 1
    print(f"  [verify] {n} items, {len(q2a)} unique queries, "
          f"0 ambiguous (every query -> exactly one answer).")


# --------------------------------------------------------------------------- #
# Cross-task batched generation
# --------------------------------------------------------------------------- #

def generate_all_tasks(args, topics, backend, novelty) -> List[List[Dict]]:
    """Fill every task to `items_per_task` valid, globally-unique, (optionally)
    novel items, batching prompts ACROSS tasks so the GPU stays saturated.

    Each round builds up to `batch_size` prompts by round-robining over the
    still-incomplete tasks (each capped at the prompts it still needs), runs them
    as one generate() call, then validates / dedups globally / novelty-filters
    and distributes the survivors back to their tasks.
    """
    n_tasks = args.n_tasks
    n_items = args.items_per_task
    batch_size = backend.parallel_prompts
    used_entities: set = set()
    seen_questions: set = set()
    kept: List[List[Dict]] = [[] for _ in range(n_tasks)]
    rounds_used = [0] * n_tasks

    def remaining(t: int) -> int:
        return n_items - len(kept[t])

    stalls = 0
    batch_no = 0
    prompts_used = 0        # running totals -> estimate usable items per prompt
    survivors_total = 0
    while any(remaining(t) > 0 for t in range(n_tasks)):
        # Estimate yield (validated, unique, novel items per prompt) from history.
        # Before any history (batch 1), assume a conservative ~1/3 of requested
        # items survive, so the first batch provisions enough to actually COMPLETE
        # its front task(s) rather than landing just short. From batch 2 on this
        # is the measured yield and self-corrects to the model's real rate.
        yield_pp = (survivors_total / prompts_used) if prompts_used else args.items_per_prompt / 3.0
        yield_pp = max(yield_pp, 0.25)
        # Depth-first fill: provision each FRONT task to its remaining need so
        # tasks COMPLETE steadily — instead of every task inching up together
        # (which shows 0 done for many batches and starves later tasks).
        reqs: List[Tuple[int, int]] = []  # (task_id, seed)
        for t in range(n_tasks):
            if len(reqs) >= batch_size:
                break
            if remaining(t) <= 0:
                continue
            want = max(1, math.ceil(remaining(t) / yield_pp * args.oversample))
            for _ in range(want):
                if len(reqs) >= batch_size:
                    break
                rounds_used[t] += 1
                reqs.append((t, args.seed + t * 100_000 + rounds_used[t] * 97))
        if not reqs:
            break

        # ── Generate the whole batch in one shot ──
        results = backend.generate(
            [(topics[t]["name"], args.items_per_prompt, seed) for (t, seed) in reqs])
        prompts_used += len(reqs)

        # ── Validate + global dedup ──
        fresh: List[Tuple[int, Dict]] = []
        for (t, _seed), cands in zip(reqs, results):
            if remaining(t) <= 0:
                continue
            for raw in cands:
                cand = _clean_candidate(raw, args.max_answer_words)
                if cand is None:
                    continue
                ekey, qkey = _norm(cand["entity"]), _norm(cand["question"])
                if ekey in used_entities or qkey in seen_questions:
                    continue
                used_entities.add(ekey)
                seen_questions.add(qkey)
                fresh.append((t, cand))

        # ── Novelty filter across the whole batch at once ──
        if novelty is not None and fresh:
            flags = novelty.is_novel([c for _t, c in fresh])
            dropped = sum(1 for ok in flags if not ok)
            fresh = [(t, c) for (t, c), ok in zip(fresh, flags) if ok]
            if dropped:
                print(f"    novelty filter dropped {dropped} already-known items")

        survivors_total += len(fresh)

        # ── Distribute survivors to their tasks ──
        added = 0
        for t, cand in fresh:
            if remaining(t) <= 0:
                continue
            kept[t].append(_build_item(t, topics[t], len(kept[t]), cand))
            added += 1

        batch_no += 1
        done = sum(1 for t in range(n_tasks) if remaining(t) <= 0)
        total = sum(len(k) for k in kept)
        print(f"  batch {batch_no}: {len(reqs)} prompts -> +{added} items  "
              f"[{done}/{n_tasks} tasks done, {total}/{n_tasks * n_items} items]")

        if added == 0:
            stalls += 1
            if stalls >= args.max_rounds:
                raise RuntimeError(
                    f"Generation stalled: {stalls} consecutive batches produced no "
                    f"new items ({total}/{n_tasks * n_items} collected). Raise "
                    f"--oversample/--max_rounds, relax/disable the novelty filter, "
                    f"or lower --items_per_task.")
        else:
            stalls = 0

    return kept


def generate_dataset(args) -> None:
    topics = _load_topics(args.spec)
    if args.n_tasks > len(topics):
        raise ValueError(
            f"Spec has {len(topics)} topics; cannot make {args.n_tasks} tasks. "
            f"Add more entries to {args.spec}.")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.backend == "qwen":
        backend = QwenGenerator(
            args.generator_model, args.gen_temperature, args.gen_top_p,
            args.max_new_tokens, args.parallel_prompts,
            mem_frac=args.gen_mem_frac, hard_cap=args.max_parallel_prompts)
    else:
        print("  Using offline template backend (smoke test; not real knowledge).")
        backend = TemplateBackend(args.seed)
        if args.parallel_prompts and args.parallel_prompts > 0:
            backend.parallel_prompts = args.parallel_prompts

    novelty = None
    if args.novelty_filter_model:
        novelty = NoveltyFilter(args.novelty_filter_model, args.novelty_batch_size)

    all_items = generate_all_tasks(args, topics, backend, novelty)

    manifest = {
        "dataset_type": "llm_qa_memorization",
        "n_tasks": args.n_tasks,
        "items_per_task": args.items_per_task,
        "repeat": args.repeat,
        "seed": args.seed,
        "spec": os.path.basename(args.spec),
        "backend": args.backend,
        "generator_model": args.generator_model if args.backend == "qwen" else None,
        "novelty_filter_model": args.novelty_filter_model,
        "parallel_prompts": backend.parallel_prompts,
        "items_per_prompt": args.items_per_prompt,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": "Question: {query}\nAnswer:",
        "tasks": [],
    }

    for t in range(args.n_tasks):
        items = all_items[t]
        texts = make_train_texts(
            items, args.repeat, random.Random(args.seed + t * 1009 + 7))

        task_dir = os.path.join(args.output_dir, f"task_{t}")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "train_items.json"), "w") as f:
            json.dump(items, f, separators=(", ", ": "))
        with open(os.path.join(task_dir, "train_texts.json"), "w") as f:
            json.dump(texts, f, separators=(", ", ": "))
        with open(os.path.join(task_dir, "test_items.json"), "w") as f:
            json.dump(items, f, separators=(", ", ": "))

        manifest["tasks"].append({
            "task_id": t, "topic": topics[t]["id"], "topic_name": topics[t]["name"],
            "n_train": len(items), "n_test": len(items), "n_train_texts": len(texts),
        })
        if t < 2:
            ex = items[0]
            print(f"Task {t} [{topics[t]['id']}]: e.g. Q={ex['query']!r} A={ex['answer']!r}")

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nVerifying disambiguation across {args.n_tasks} tasks ...")
    verify_no_global_ambiguity(all_items)
    print(f"Saved LLM-knowledge QA dataset to {args.output_dir}/")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate LLM-synthesized novel-knowledge QA data for CL.")
    p.add_argument("--n_tasks", type=int, default=100)
    p.add_argument("--items_per_task", type=int, default=100)
    p.add_argument("--repeat", type=int, default=1,
                   help="Times each Q/A string appears in train_texts.json.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="data/synthetic_qa/llm_qa")
    p.add_argument("--spec", type=str, default=DEFAULT_SPEC,
                   help="Path to llm_qa_topics.json")

    p.add_argument("--backend", choices=["qwen", "template"], default="qwen",
                   help="qwen: local HF model (real). template: offline smoke test.")
    p.add_argument("--generator_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--gen_temperature", type=float, default=0.9)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=1536,
                   help="Cap per completion; ~items_per_prompt small JSON objects.")
    p.add_argument("--items_per_prompt", type=int, default=10,
                   help="Candidate items requested per LLM completion. Smaller -> "
                        "more prompts -> larger cross-task batch.")
    p.add_argument("--parallel_prompts", type=int, default=0,
                   help="Prompts per GPU batch (pooled across tasks). 0 = auto-size "
                        "to free VRAM (recommended). Set >0 to pin a fixed batch.")
    p.add_argument("--max_parallel_prompts", type=int, default=64,
                   help="Hard cap for the auto-sized batch.")
    p.add_argument("--gen_mem_frac", type=float, default=0.7,
                   help="Target fraction of free VRAM for the auto-sized batch.")
    p.add_argument("--oversample", type=float, default=1.3,
                   help="Safety margin on the measured items-per-prompt yield when "
                        "provisioning each task's prompts (absorbs variance so tasks "
                        "finish in ~1 batch instead of many top-up rounds).")
    p.add_argument("--max_rounds", type=int, default=60,
                   help="Abort after this many consecutive batches with no new items.")
    p.add_argument("--max_answer_words", type=int, default=8)

    p.add_argument("--novelty_filter_model", type=str, default=None,
                   help="If set (e.g. the CL base model Qwen/Qwen3-4B-Base), drop "
                        "any generated item this model already answers correctly, "
                        "guaranteeing the kept facts are novel.")
    p.add_argument("--novelty_batch_size", type=int, default=64)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
