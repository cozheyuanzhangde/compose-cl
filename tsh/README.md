# Task-Level Successive Halving

Task-Level Successive Halving (TSH) seeks preliminary evidence for the
composition hypothesis by allocating progressively longer task horizons to
surviving methods. Unlike conventional
successive halving, its resource is the number of sequential tasks rather than
the number of optimization iterations.

The paper search crosses four dimensions:

- weight anchor: none, online EWC, or SI.
- function anchor: none or self-distillation with weight 1 or 3.
- data anchor: none or replay with the four combinations of loss weight
  `{0.5, 0.75}` and generation temperature `{1.0, 1.5}`.
- low-rank allocation: shared LoRA or merged LoRA.

This gives `3 × 3 × 5 × 2 = 90` initial configurations. Every score is final
retention averaged over seeds 41, 42, and 43 on the fixed development task
order (`task_order_seed=1234`). The funnel is:

| Horizon | Evaluated | Continue |
|---:|---:|---:|
| 10 tasks | 90 | 45 |
| 20 tasks | 45 | 23 |
| 50 tasks | 23 | 10 |
| 100 tasks | 10 | final |

O-LoRA and sequential OSRM are evaluated separately after selection because
their retained state grows linearly with the number of tasks.

Run a dry plan before allocating GPUs:

```bash
python tsh/run.py tsh/configs/symbol_qa.yaml --dry_run
```

Run the paper search on one or more visible GPUs:

```bash
PY="$(command -v python)" \
  python tsh/run.py tsh/configs/symbol_qa.yaml --gpus 0,1,2,3 --per_gpu 1
```

Replace the config with `llm_qa.yaml` or `real_qa.yaml` for the other datasets.
Completed cells are reused, promoted configurations continue from the previous
rung, and an interrupted cell resumes at its last task boundary. Outputs are
written below `results/tsh/<dataset>/`:

- `paper/leaderboard.md`, `trace.json`, and `progress.html` record the funnel.
- `cells/paper/<configuration>/rung_<horizon>/` contains checkpoints, logs, and
  evaluation results.
- `cell.json` records the exact configuration and prevents accidental reuse
  under mismatched hyperparameters.

The final evaluation uses the default on-disk task order.
`experiments/run_final.py` deliberately omits `--task_order_seed`.
