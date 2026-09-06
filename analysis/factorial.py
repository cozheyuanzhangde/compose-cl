#!/usr/bin/env python3
r"""2^4 factorial ANOVA over {SI, SD, replay, merge}.

Reads canonical final runs (per-seed values live in
`metrics_summary.json` -> configs[<cfg>]['agg'][<metric>]['values']) and fits a
saturated 2^4 model by Yates/contrast algebra:

    effect(S) = mean over the 16 cells of (+/-1)^{|S \ active|} * cell_mean * 2
              = mean[factor on] - mean[factor off]        (for a main effect)

With a balanced design and n seeds per cell, every effect has the same standard
error, so F = SS_effect / MS_error with 1 and 16(n-1) df. Cell means come from
per-seed values, and MS_error is the pooled within-cell variance -- i.e. seed
noise is the error term, which is the right reference for "would another seed
have reversed this?".

Usage:  python analysis/factorial.py [--metric final_acc]
Outputs (analysis/outputs by default):
    anova_<stream>.md          per-stream effect table
    anova_all_streams.md       side-by-side comparison + the SI x merge ladder
    anova_all_streams.json     machine-readable
"""
import argparse
import itertools
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(REPO, "results", "final")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
STREAMS = ["symbol_qa", "llm_qa", "real_qa"]
FACTORS = ["si", "sd", "replay", "merge"]     # canonical order for cell names


def cell_name(active):
    """Config directory name for a subset of active factors ('vanilla' if none)."""
    parts = [f for f in FACTORS if f in active]
    return "_".join(parts) if parts else "vanilla"


def load_cells(stream, metric, results_root):
    """-> {frozenset(active factors): np.array of per-seed values (in %)}"""
    path = os.path.join(results_root, stream, "metrics_summary.json")
    cfgs = json.load(open(path))["configs"]
    cells, missing = {}, []
    for r in range(len(FACTORS) + 1):
        for active in itertools.combinations(FACTORS, r):
            name = cell_name(active)
            entry = cfgs.get(name)
            if entry is None or metric not in entry.get("agg", {}):
                missing.append(name)
                continue
            cells[frozenset(active)] = np.array(
                entry["agg"][metric]["values"], dtype=float) * 100.0
    return cells, missing


def fit(cells):
    """Saturated 2^4 ANOVA. Returns (terms, diagnostics)."""
    keys = sorted(cells, key=lambda s: (len(s), sorted(s)))
    n_per = {len(cells[k]) for k in keys}
    if len(n_per) != 1:
        raise ValueError(f"unbalanced seed counts: {n_per}")
    n = n_per.pop()
    N = len(keys) * n

    means = {k: cells[k].mean() for k in keys}
    # pooled within-cell (seed) variance
    ss_err = sum(((cells[k] - means[k]) ** 2).sum() for k in keys)
    df_err = len(keys) * (n - 1)
    ms_err = ss_err / df_err

    grand = np.mean([means[k] for k in keys])
    ss_tot = sum(((cells[k] - grand) ** 2).sum() for k in keys)

    terms = []
    for r in range(1, len(FACTORS) + 1):
        for subset in itertools.combinations(FACTORS, r):
            S = frozenset(subset)
            # contrast: +1 if |S ∩ active| even ... standard 2^k sign algebra
            signs = {k: (-1) ** len(S - k) for k in keys}
            # effect = (mean at +1) - (mean at -1), each averaged over 8 cells
            plus = [means[k] for k in keys if signs[k] > 0]
            minus = [means[k] for k in keys if signs[k] < 0]
            effect = np.mean(plus) - np.mean(minus)
            # SS for a 2^k contrast with n reps: N * effect^2 / 4
            ss = N * effect ** 2 / 4.0
            F = ss / ms_err
            terms.append({
                "term": " x ".join(subset), "order": r, "effect": effect,
                "ss": ss, "F": F, "p": f_sf(F, 1, df_err),
                "pct_var": 100.0 * ss / ss_tot,
            })
    terms.sort(key=lambda d: -abs(d["effect"]))
    diag = {"n_seeds": n, "N": N, "df_err": df_err,
            "rmse": float(np.sqrt(ms_err)), "grand_mean": float(grand),
            "ss_total": float(ss_tot),
            "pct_var_explained": 100.0 * sum(t["ss"] for t in terms) / ss_tot,
            # SE of an effect = 2*sigma/sqrt(N); a "1 sigma" yardstick for readers
            "se_effect": float(2 * np.sqrt(ms_err / N))}
    return terms, diag


def f_sf(F, df1, df2):
    """Upper-tail p for F(df1, df2) via the incomplete beta (no scipy needed)."""
    if F <= 0:
        return 1.0
    x = df2 / (df2 + df1 * F)
    return betainc(df2 / 2.0, df1 / 2.0, x)


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a,b) by continued fraction (NR 6.4)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (lgamma(a + b) - lgamma(a) - lgamma(b)
             + a * np.log(x) + b * np.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return np.exp(lbeta) * betacf(a, b, x) / a
    return 1.0 - np.exp(lbeta) * betacf(b, a, 1 - x) / b


def betacf(a, b, x, itmax=300, eps=3e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def lgamma(z):
    from math import lgamma as _lg
    return _lg(z)


def md_table(rows, cols, headers, fmts):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(
            f.format(r[c]) if not isinstance(r[c], str) else r[c]
            for c, f in zip(cols, fmts)) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="final_acc",
                    help="key under agg/ (final_acc, diag, W10, W50, ...)")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    results_root = os.path.abspath(args.results_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    report, blob = [], {"metric": args.metric, "source": results_root, "streams": {}}
    report.append(f"# 2^4 factorial ANOVA over {{SI, SD, replay, merge}} "
                  f"— metric `{args.metric}`\n")
    report.append(f"Source: `{results_root}/<stream>/metrics_summary.json` "
                  "(per-seed values). Error term = pooled within-cell seed "
                  "variance. Effect = mean[on] − mean[off] in percentage "
                  "points.\n")

    per_stream = {}
    for stream in STREAMS:
        cells, missing = load_cells(stream, args.metric, results_root)
        if missing:
            report.append(f"\n> **{stream}: missing cells {missing}** — skipped.\n")
            continue
        terms, diag = fit(cells)
        per_stream[stream] = (cells, terms, diag)
        blob["streams"][stream] = {"diag": diag, "terms": terms}

        lines = [f"# {stream} — 2^4 ANOVA (`{args.metric}`)\n",
                 f"16 cells x {diag['n_seeds']} seeds = N={diag['N']}; "
                 f"error df={diag['df_err']}; RMSE={diag['rmse']:.2f} pp; "
                 f"grand mean={diag['grand_mean']:.2f}; "
                 f"SE(effect)={diag['se_effect']:.2f} pp.\n"]
        lines.append(md_table(
            terms, ["term", "effect", "pct_var", "F", "p"],
            ["term", "effect (pp)", "%var", "F", "p"],
            ["{}", "{:+.2f}", "{:.1f}", "{:.1f}", "{:.2g}"]))
        lines.append("\n\n## Cell means (mean ± sd over seeds)\n")
        cm = [{"cfg": cell_name(k),
               "m": float(cells[k].mean()), "s": float(cells[k].std(ddof=1))}
              for k in sorted(cells, key=lambda s: -cells[s].mean())]
        lines.append(md_table(cm, ["cfg", "m", "s"],
                              ["config", "mean", "sd"],
                              ["{}", "{:.1f}", "{:.1f}"]))
        with open(os.path.join(output_dir, f"anova_{stream}.md"), "w") as handle:
            handle.write("\n".join(lines) + "\n")

    # ---- cross-stream comparison ----
    report.append("\n## Main effects and key interactions across streams\n")
    order = ["replay", "merge", "sd", "si", "replay x merge", "si x merge",
             "sd x replay", "si x sd", "sd x merge", "si x replay"]
    rows = []
    for t in order:
        row = {"term": t}
        for s in STREAMS:
            if s not in per_stream:
                row[s] = "--"
                continue
            hit = next((x for x in per_stream[s][1] if x["term"] == t), None)
            row[s] = (f"{hit['effect']:+.2f}"
                      + ("" if hit["p"] < 0.05 else " ns")) if hit else "--"
        rows.append(row)
    report.append(md_table(rows, ["term"] + STREAMS,
                           ["term"] + STREAMS, ["{}"] * (1 + len(STREAMS))))

    with open(os.path.join(output_dir, "anova_all_streams.md"), "w") as handle:
        handle.write("\n".join(report) + "\n")
    with open(os.path.join(output_dir, "anova_all_streams.json"), "w") as handle:
        json.dump(blob, handle, indent=1)
    print(f"wrote ANOVA reports to {output_dir}/")


if __name__ == "__main__":
    main()
