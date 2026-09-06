"""Unit test for core/mem_log.py (CPU-only; no GPU required).

Checks the contract that train.py / core/training.py rely on:
  * pre-init global logger is a no-op that never raises;
  * init_mem_log is idempotent and survives an unwritable path;
  * task_train_start/end emit the expected boundary rows
    (N starts, N ends, N-1 inter_task_end) plus sampler rows;
  * every row carries RSS; custom mark() kwargs land in the row.

Run:  python -m pytest tests/test_mem_log.py -q
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mem_log import init_mem_log, get_mem_log, _NoopLogger  # noqa: E402


def test_mem_log():
    # pre-init: noop logger, calls never crash
    assert isinstance(get_mem_log(), _NoopLogger)
    get_mem_log().task_train_start()
    get_mem_log().mark("x")

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "mem_log.jsonl")
        lg = init_mem_log(p, interval=0.2, meta={"script": "unit"})
        assert get_mem_log() is lg and lg.enabled
        for _ in range(3):
            lg.task_train_start()
            lg.task_train_end()
        time.sleep(0.5)  # let the sampler thread write a few rows
        lg.mark("custom", foo=1)
        lg.close()

        rows = [json.loads(line) for line in open(p)]
        ev = [r["event"] for r in rows]
        assert ev[0] == "meta"
        assert ev.count("task_train_start") == 3
        assert ev.count("task_train_end") == 3
        assert ev.count("inter_task_end") == 2
        assert "sample" in ev and rows[-1]["event"] == "final"
        assert all("rss_mb" in r for r in rows)
        assert [r for r in rows if r["event"] == "custom"][0]["foo"] == 1
        # idempotent: second init returns the live logger unchanged
        assert init_mem_log("/nonexistent/x.jsonl") is lg

if __name__ == "__main__":
    test_mem_log()
    print("test_mem_log: OK")
