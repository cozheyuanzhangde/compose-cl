"""Lightweight GPU / RAM usage logger (JSONL) for train/eval runs.

Motivation: methods differ wildly in memory footprint (LwF keeps a frozen
teacher copy, generative replay snapshots a second generator model, the
3-stack keeps both, O-LoRA's frozen A/B registries grow with #tasks, fold
merging materializes full-weight saves) and we need real numbers per
(model, dataset, method, #tasks) combination to plan large-scale runs.

Design constraints:
  * PURELY OBSERVATIONAL — only reads CUDA memory counters and
    /proc/self/status and appends JSONL rows. It never touches the RNG,
    never allocates GPU memory, so enabling it keeps training
    bit-identical (regression-gate safe).
  * NEVER crashes the run — every write is wrapped; failures degrade to
    silence, not exceptions.
  * Near-zero overhead — a daemon thread samples every `interval` seconds
    (a handful of counter reads), plus one boundary row per task.

Wiring (already done in train.py / evaluate.py, opt-out via --no_mem_log):
    from core.mem_log import init_mem_log, get_mem_log
    init_mem_log(os.path.join(args.output_dir, "mem_log.jsonl"),
                 meta={"argv": sys.argv[1:]})
    ...
    get_mem_log().mark("some_event", extra_key=...)   # optional manual marks

Per-task boundaries are emitted automatically from core/training.py
(train_one_task / train_one_task_olora / train_one_task_replay):
  * `task_train_start` — resets the CUDA peak counters, so the matching
    `task_train_end` row's max_alloc_mb is the training peak of THAT task;
  * `inter_task_end` (written just before the next task's start) — peak
    since the previous task's start, i.e. it additionally covers the
    post-task fold/merge + checkpoint save and the next task's teacher
    snapshot / replay generation;
  * `sample` rows every `interval` seconds give the full timeline.

Row schema (MB everywhere):
  {"event", "t" (unix), "task", "alloc_mb", "reserved_mb", "max_alloc_mb",
   "max_reserved_mb", "rss_mb", "rss_peak_mb", ...event-specific extras}
The first row is `{"event": "meta", ...}` with argv / device info.

"""
from __future__ import annotations

import json
import threading
import time

import torch


class _NoopLogger:
    """Inactive stand-in so library code can call hooks unconditionally."""
    enabled = False

    def mark(self, event, **kw):
        pass

    def task_train_start(self, task=None):
        pass

    def task_train_end(self, task=None):
        pass

    def set_task_base(self, next_task: int):
        pass

    def close(self):
        pass


class MemLogger:
    enabled = True

    def __init__(self, path: str, interval: float = 10.0, meta: dict = None):
        self.path = path
        self.interval = interval
        self._lock = threading.Lock()
        self._task = None          # current task index (auto-incremented)
        self._auto_task = -1
        self._stop = threading.Event()
        row = {"event": "meta", **(meta or {})}
        if torch.cuda.is_available():
            try:
                prop = torch.cuda.get_device_properties(0)
                row["device_name"] = prop.name
                row["device_total_mb"] = round(prop.total_memory / 2**20)
            except Exception:
                pass
        self._write({**row, **self._common()})
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="mem_log_sampler")
        self._thread.start()

    # ── internals ──────────────────────────────────────────────────────
    @staticmethod
    def _rss_mb():
        """VmRSS / VmHWM from /proc/self/status (Linux), in MB."""
        try:
            out = {}
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        out["rss_mb"] = int(line.split()[1]) // 1024
                    elif line.startswith("VmHWM:"):
                        out["rss_peak_mb"] = int(line.split()[1]) // 1024
            return out
        except Exception:
            return {}

    def _common(self):
        row = {"t": round(time.time(), 1), "task": self._task}
        if torch.cuda.is_available():
            try:
                mb = 2**20
                row["alloc_mb"] = round(torch.cuda.memory_allocated() / mb)
                row["reserved_mb"] = round(torch.cuda.memory_reserved() / mb)
                row["max_alloc_mb"] = round(torch.cuda.max_memory_allocated() / mb)
                row["max_reserved_mb"] = round(torch.cuda.max_memory_reserved() / mb)
            except Exception:
                pass
        row.update(self._rss_mb())
        return row

    def _write(self, row):
        try:
            with self._lock, open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass  # logging must never crash the run

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._write({"event": "sample", **self._common()})

    # ── public API ─────────────────────────────────────────────────────
    def mark(self, event: str, **kw):
        self._write({"event": event, **self._common(), **kw})

    def task_train_start(self, task=None):
        """Called at the top of every train_one_task* invocation."""
        if self._task is not None:
            # Peak since the PREVIOUS task's reset: its training plus the
            # post-task fold/save and this task's teacher/replay-gen setup.
            self.mark("inter_task_end")
        self._auto_task += 1
        self._task = self._auto_task if task is None else task
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self.mark("task_train_start")

    def task_train_end(self, task=None):
        """Called at the bottom of every train_one_task* invocation.

        max_alloc_mb in this row == the task's TRAINING peak (counters were
        reset at task_train_start)."""
        self.mark("task_train_end")

    def set_task_base(self, next_task: int):
        """Align the auto-incremented task counter after a mid-run resume so
        appended rows keep stream-global task indices (the jsonl is opened in
        append mode, so the pre-kill rows are still there). Emits a 'resume'
        tombstone row first: the killed task's earlier partial rows precede it,
        so a reader can discard rows for task >= next_task before the last
        'resume'/'meta' marker when summing wall-clock."""
        self.mark("resume", next_task=int(next_task))
        self._auto_task = int(next_task) - 1

    def close(self):
        self._stop.set()
        self.mark("final")


_logger = _NoopLogger()


def init_mem_log(path: str, interval: float = 10.0, meta: dict = None):
    """Activate the global logger (idempotent; safe to call once per proc)."""
    global _logger
    try:
        if isinstance(_logger, _NoopLogger):
            _logger = MemLogger(path, interval=interval, meta=meta)
    except Exception:
        _logger = _NoopLogger()
    return _logger


def get_mem_log():
    return _logger
