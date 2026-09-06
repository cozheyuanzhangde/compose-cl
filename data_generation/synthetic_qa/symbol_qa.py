"""
Symbol-QA generator: pure-memorization task sequences for continual learning.

The most "fully synthetic" tier of the memorization suite (no semantics at all):
arbitrary symbol associations the model can only memorize, never derive.

Paradigm
--------
Every task is an ARBITRARY fixed codebook — a batch of random injective lookups
from a short symbol "key" to a short symbol "value", drawn from ONE shared rule
(a single alphabet + fixed key/value lengths) defined in `symbol_qa_types.json`:

    aB3xKq -> qW8z       (mixed letters+digits, task 0)
    7mZ0pL -> H4tv       (same rule,            task 37)

There is NO rule relating a key to its value, so nothing can be generalized:
the only way to answer is to have memorized the exact pair. This is the
deliberate opposite of `generate_selector_data.py`, whose tasks are
*structural rules* meant to test generalization (held-out pairs). Here:

  - Train and test are IDENTICAL (the same key->value pairs are shown at
    train time and queried at test time). We measure how well memorized
    pairs survive training on later tasks (catastrophic forgetting).
  - ALL tasks share the SAME rule (alphabet, key length, value length). There
    are NO per-task "types" and NO surface-form blocks: the 100 tasks are a
    HOMOGENEOUS stream of i.i.d. memorization batches. A task is just a fresh
    batch of 100 globally-unique pairs; nothing in the surface form marks where
    one task ends and the next begins. This removes the surface-form confound of
    the earlier "10 family blocks of 10" layout, so the accuracy matrix reflects
    recency/interference forgetting rather than alphabet switches.

Disambiguation (why every test query has exactly one answer)
-----------------------------------------------------------
At test time the model is shown only `Question: <key>\nAnswer:` with NO task
label, so the global map key -> value MUST be single-valued across the whole
100-task stream. We guarantee this at generation time with a GLOBAL key
registry: every key emitted by any task is unique across all tasks (and hence
within each task too). A key therefore identifies exactly one (task, value),
and the correct answer is never ambiguous. A final verification pass asserts
this property over the assembled dataset. (Values are unconstrained — only the
key, which is what the model is queried on, must be unique.)

Output format matches `train.py` / `evaluate.py`:
  task_t/train_texts.json  - List[str], each "Question: <key>\nAnswer: <value>"
  task_t/train_items.json  - List[Dict] with id/query/answer/train_text/...
  task_t/test_items.json   - List[Dict] (identical pairs to train)
  manifest.json            - n_tasks + shared rule + per-task metadata

Usage:
    python data_generation/synthetic_qa/symbol_qa.py --n_tasks 100 --items_per_task 100
"""
from __future__ import annotations
import argparse
import json
import os
import random
from typing import Dict, List

DEFAULT_SPEC = os.path.join(os.path.dirname(__file__), "symbol_qa_types.json")


def _load_rule(spec_path: str) -> Dict:
    """Load the single shared codebook rule (alphabet + key/value lengths)."""
    with open(spec_path) as f:
        spec = json.load(f)
    rule = spec.get("rule")
    if not rule:
        raise ValueError(
            f"No 'rule' object found in {spec_path}. The symbol-QA spec now "
            f"holds ONE shared rule (single alphabet + key_len + value_len) "
            f"used for every task, not a list of per-task 'types'.")
    for field in ("id", "alphabet", "key_len", "value_len"):
        if field not in rule:
            raise ValueError(f"Rule in {spec_path} is missing '{field}'.")
    if not rule["alphabet"]:
        raise ValueError(f"Rule in {spec_path} has an empty alphabet.")
    return rule


def _rand_string(rng: random.Random, alphabet: str, length: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def generate_task_items(
    task_id: int,
    rule: Dict,
    n_items: int,
    rng: random.Random,
    used_keys: set,
    max_attempts: int = 1_000_000,
) -> List[Dict]:
    """Emit `n_items` unique key->value pairs for one codebook task.

    Keys are drawn without replacement from the shared rule's alphabet and are
    additionally checked against `used_keys` (the GLOBAL registry of every key
    ever emitted) so that no key is shared with any other task. Values are
    unconstrained — only the key must be unique, since the key is what the model
    is queried on.
    """
    alphabet = rule["alphabet"]
    key_len = int(rule["key_len"])
    value_len = int(rule["value_len"])

    items: List[Dict] = []
    attempts = 0
    while len(items) < n_items:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Task {task_id} ({rule['id']}): exhausted {max_attempts} "
                f"attempts at {len(items)}/{n_items} unique keys. Increase "
                f"key_len or the alphabet in {DEFAULT_SPEC}.")
        key = _rand_string(rng, alphabet, key_len)
        if key in used_keys:
            continue
        used_keys.add(key)
        value = _rand_string(rng, alphabet, value_len)
        i = len(items)
        train_text = f"Question: {key}\nAnswer: {value}"
        items.append({
            "id": f"task_{task_id}_item_{i}",
            "task_id": task_id,
            "category": rule["id"],
            "query": key,
            "answer": value,
            "train_text": train_text,
            "key": key,
            "value": value,
        })
    return items


def make_train_texts(items: List[Dict], n_repeat: int, rng: random.Random) -> List[str]:
    """train_texts == the exact Q/A strings used at test (memorization)."""
    texts: List[str] = []
    for item in items:
        qa = f"Question: {item['query']}\nAnswer: {item['answer']}"
        texts.extend([qa] * n_repeat)
    rng.shuffle(texts)
    return texts


def verify_no_global_ambiguity(all_items: List[List[Dict]]) -> None:
    """Assert the global query -> answer map is single-valued and well formed.

    This is the core guarantee: with no task label at test time, each query
    must determine exactly one answer across the entire task stream.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate symbol-QA memorization data for CL.")
    parser.add_argument("--n_tasks", type=int, default=100)
    parser.add_argument("--items_per_task", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1,
                        help="How many times each Q/A string appears in "
                             "train_texts.json (test_items stays deduplicated).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data/synthetic_qa/symbol_qa")
    parser.add_argument("--spec", type=str, default=DEFAULT_SPEC,
                        help="Path to symbol_qa_types.json (holds the single "
                             "shared rule used for every task).")
    args = parser.parse_args()

    rule = _load_rule(args.spec)
    alphabet = rule["alphabet"]
    key_len = int(rule["key_len"])
    value_len = int(rule["value_len"])

    # Capacity check: the GLOBAL key registry must hold every task's keys without
    # heavy rejection sampling. With one shared alphabet, the binding constraint
    # is n_tasks * items_per_task globally-unique keys (not per-task).
    key_space = len(alphabet) ** key_len
    total_keys = args.n_tasks * args.items_per_task
    if key_space < total_keys * 4:
        raise ValueError(
            f"Key space {len(alphabet)}^{key_len} = {key_space} is too small "
            f"for {total_keys} globally-unique keys "
            f"({args.n_tasks} tasks x {args.items_per_task} items). Increase "
            f"key_len or enlarge the alphabet in {args.spec}.")

    os.makedirs(args.output_dir, exist_ok=True)
    used_keys: set = set()
    all_items: List[List[Dict]] = []

    manifest = {
        "dataset_type": "symbol_qa_memorization",
        "n_tasks": args.n_tasks,
        "items_per_task": args.items_per_task,
        "repeat": args.repeat,
        "seed": args.seed,
        "spec": os.path.basename(args.spec),
        "prompt_template": "Question: {query}\nAnswer:",
        # Single shared rule for the whole stream (no per-task types / blocks).
        "rule": {
            "id": rule["id"],
            "name": rule.get("name", rule["id"]),
            "alphabet_id": rule.get("alphabet_id", "custom"),
            "key_len": key_len,
            "value_len": value_len,
        },
        "tasks": [],
    }

    for t in range(args.n_tasks):
        rng = random.Random(args.seed + t * 1009)
        items = generate_task_items(
            t, rule, args.items_per_task, rng, used_keys)
        texts = make_train_texts(
            items, args.repeat, random.Random(args.seed + t * 1009 + 7))
        all_items.append(items)

        # Memorization: test items == train items (identical pairs).
        task_dir = os.path.join(args.output_dir, f"task_{t}")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "train_items.json"), "w") as f:
            json.dump(items, f, separators=(", ", ": "))
        with open(os.path.join(task_dir, "train_texts.json"), "w") as f:
            json.dump(texts, f, separators=(", ", ": "))
        with open(os.path.join(task_dir, "test_items.json"), "w") as f:
            json.dump(items, f, separators=(", ", ": "))

        # Per-task metadata mirrors the shared rule (kept for back-compat with
        # tooling that reads task["alphabet_id"]/["category"]; all tasks are the
        # same kind now, so these are identical across tasks).
        manifest["tasks"].append({
            "task_id": t,
            "type_id": rule["id"],
            "type_name": rule.get("name", rule["id"]),
            "alphabet_id": rule.get("alphabet_id", "custom"),
            "category": rule["id"],
            "key_len": key_len,
            "value_len": value_len,
            "n_train": len(items),
            "n_test": len(items),
            "n_train_texts": len(texts),
        })
        if t < 2:
            ex = items[0]
            print(f"Task {t} [{rule['id']}]: {len(items)} pairs  "
                  f"e.g. {ex['query']!r} -> {ex['answer']!r}")

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nVerifying disambiguation across {args.n_tasks} tasks "
          f"(shared rule {rule['id']}) ...")
    verify_no_global_ambiguity(all_items)
    print(f"Saved symbol-QA dataset to {args.output_dir}/")


if __name__ == "__main__":
    main()
