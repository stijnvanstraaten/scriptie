#!/usr/bin/env python3
"""
analyze_results.py

Reads results_perplexity.csv from your stego evaluation and produces:
- Descriptive tables (capacity + success + perplexity)
- Statistical tests:
    * Kruskal–Wallis on capacity across modes
    * Pairwise Mann–Whitney U with Holm correction (capacity)
    * Success-rate differences across modes via chi-square test
      + pairwise Fisher exact tests with Holm correction (success)
    * Correlations between capacity and ΔPPL (Pearson + Spearman) per mode
- Plots:
    * trade-off scatter (capacity vs ΔPPL)
    * bar charts: mean capacity, success rate, mean ΔPPL
- Outputs:
    * CSV tables to an output folder
    * LaTeX table snippets you can paste into your thesis
    * PNG figures

Usage:
  python analyze_results.py /path/to/results_perplexity.csv --outdir out_results
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None


MODES_ORDER = ["NOUN", "VERB", "ADJ", "ANY"]


def holm_correction(pvals: List[float]) -> np.ndarray:
    """Holm correction (fallback if statsmodels isn't installed)."""
    p = np.array(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty_like(ranked)
    for i in range(m):
        adj[i] = min(1.0, (m - i) * ranked[i])
    # enforce monotonicity
    for i in range(1, m):
        adj[i] = max(adj[i], adj[i - 1])
    out = np.empty_like(adj)
    out[order] = adj
    return out


def format_float(x, nd=2):
    if pd.isna(x):
        return "--"
    return f"{x:.{nd}f}"


def latex_capacity_table(cap_stats: pd.DataFrame) -> str:
    lines = []
    for mode in MODES_ORDER:
        row = cap_stats[cap_stats["mode"] == mode]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"{mode} & {format_float(r['mean'],2)} & {format_float(r['median'],1)} & {format_float(r['IQR'],1)} \\\\"
        )
    return "\n".join(lines)


def latex_success_table(succ: pd.DataFrame) -> str:
    lines = []
    for mode in MODES_ORDER:
        row = succ[succ["mode"] == mode]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"{mode} & {int(r['successful'])} / {int(r['total'])} & {format_float(r['success_rate_pct'],1)} \\\\"
        )
    return "\n".join(lines)


def latex_ppl_table(ppl_stats: pd.DataFrame) -> str:
    lines = []
    for mode in MODES_ORDER:
        row = ppl_stats[ppl_stats["mode"] == mode]
        if row.empty:
            # e.g., ADJ if recall had 0 successes
            lines.append(f"{mode} & -- & -- & -- \\\\")
            continue
        r = row.iloc[0]
        lines.append(
            f"{mode} & {format_float(r['ppl_orig_mean'],2)} & {format_float(r['ppl_stego_mean'],2)} & {format_float(r['delta_ppl_mean'],2)} \\\\"
        )
    return "\n".join(lines)


def ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Path to results_perplexity.csv")
    parser.add_argument("--outdir", type=str, default="out_results", help="Output directory")
    parser.add_argument("--modes", type=str, default="NOUN,VERB,ADJ,ANY", help="Comma-separated modes")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    csv_path = Path(args.csv_path)

    df = pd.read_csv(csv_path)

    # Basic cleanup / consistency
    df["mode"] = df["mode"].astype(str).str.upper()
    modes = [m.strip().upper() for m in args.modes.split(",") if m.strip()]
    df = df[df["mode"].isin(modes)].copy()

    # Some runs have NaN for ppl_stego/delta_ppl when success==0
    # We'll use success==1 for perplexity comparisons.
    df["success"] = df["success"].fillna(0).astype(int)
    df["fits"] = df["fits"].fillna(0).astype(int)

    # ---------- TABLE 1: Capacity descriptive stats ----------
    cap_stats = (
        df.groupby("mode")["capacity_bits"]
        .agg(
            mean="mean",
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    cap_stats["IQR"] = cap_stats["q3"] - cap_stats["q1"]
    cap_stats = cap_stats[["mode", "mean", "median", "q1", "q3", "IQR"]]

    # ---------- TABLE 2: Success rates ----------
    succ = (
        df.groupby("mode")
        .agg(total=("success", "count"), successful=("success", "sum"))
        .reset_index()
    )
    succ["success_rate_pct"] = 100.0 * succ["successful"] / succ["total"]

    # ---------- TABLE 3: Perplexity stats (only successes) ----------
    df_succ = df[df["success"] == 1].copy()
    ppl_stats = (
        df_succ.groupby("mode")
        .agg(
            ppl_orig_mean=("ppl_content", "mean"),
            ppl_stego_mean=("ppl_stego", "mean"),
            delta_ppl_mean=("delta_ppl", "mean"),
        )
        .reset_index()
    )

    # Save tables
    cap_stats.to_csv(outdir / "table_capacity_stats.csv", index=False)
    succ.to_csv(outdir / "table_success_rates.csv", index=False)
    ppl_stats.to_csv(outdir / "table_ppl_stats_success_only.csv", index=False)

    # ---------- STATS: Kruskal–Wallis on capacity ----------
    cap_groups = [df[df["mode"] == m]["capacity_bits"].dropna().values for m in modes]
    kw_stat, kw_p = stats.kruskal(*cap_groups)

    # Pairwise Mann–Whitney U on capacity with Holm correction
    pair_rows = []
    pvals = []
    pairs = []
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            m1, m2 = modes[i], modes[j]
            a = df[df["mode"] == m1]["capacity_bits"].dropna().values
            b = df[df["mode"] == m2]["capacity_bits"].dropna().values
            # two-sided Mann–Whitney
            u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            pairs.append((m1, m2))
            pvals.append(p)
            pair_rows.append({"mode1": m1, "mode2": m2, "U": u_stat, "p_raw": p})

    if multipletests is not None:
        p_adj = multipletests(pvals, method="holm")[1]
    else:
        p_adj = holm_correction(pvals)

    for k, adj in enumerate(p_adj):
        pair_rows[k]["p_holm"] = adj

    pairwise_capacity = pd.DataFrame(pair_rows).sort_values("p_holm")
    pairwise_capacity.to_csv(outdir / "pairwise_capacity_mannwhitney_holm.csv", index=False)

    # ---------- STATS: Success differences ----------
    # Global chi-square test on 2xK table (success/fail by mode)
    # Build contingency: rows=mode, cols=[success, fail]
    contingency = []
    for m in modes:
        total = int(succ.loc[succ["mode"] == m, "total"].iloc[0])
        successful = int(succ.loc[succ["mode"] == m, "successful"].iloc[0])
        fail = total - successful
        contingency.append([successful, fail])
    chi2_stat, chi2_p, chi2_dof, _ = stats.chi2_contingency(contingency)

    # Pairwise Fisher exact on success (2x2) with Holm correction
    fisher_rows = []
    fisher_pvals = []
    fisher_pairs = []
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            m1, m2 = modes[i], modes[j]
            tot1 = int(succ.loc[succ["mode"] == m1, "total"].iloc[0])
            suc1 = int(succ.loc[succ["mode"] == m1, "successful"].iloc[0])
            fail1 = tot1 - suc1
            tot2 = int(succ.loc[succ["mode"] == m2, "total"].iloc[0])
            suc2 = int(succ.loc[succ["mode"] == m2, "successful"].iloc[0])
            fail2 = tot2 - suc2
            table_2x2 = np.array([[suc1, fail1], [suc2, fail2]])
            # two-sided fisher
            _, p = stats.fisher_exact(table_2x2, alternative="two-sided")
            fisher_pairs.append((m1, m2))
            fisher_pvals.append(p)
            fisher_rows.append({"mode1": m1, "mode2": m2, "p_raw": p})

    if multipletests is not None:
        fisher_adj = multipletests(fisher_pvals, method="holm")[1]
    else:
        fisher_adj = holm_correction(fisher_pvals)

    for k, adj in enumerate(fisher_adj):
        fisher_rows[k]["p_holm"] = adj

    pairwise_success = pd.DataFrame(fisher_rows).sort_values("p_holm")
    pairwise_success.to_csv(outdir / "pairwise_success_fisher_holm.csv", index=False)

    # ---------- Correlations: capacity vs delta_ppl per mode (success only) ----------
    corr_rows = []
    for m in modes:
        sub = df_succ[df_succ["mode"] == m]
        if len(sub) < 3:
            corr_rows.append(
                {"mode": m, "n": len(sub), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
            )
            continue
        pear_r, pear_p = stats.pearsonr(sub["capacity_bits"], sub["delta_ppl"])
        spear_r, spear_p = stats.spearmanr(sub["capacity_bits"], sub["delta_ppl"])
        corr_rows.append({"mode": m, "n": len(sub), "pearson_r": pear_r, "pearson_p": pear_p, "spearman_r": spear_r, "spearman_p": spear_p})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(outdir / "correlations_capacity_vs_delta_ppl.csv", index=False)

    # ---------- PLOTS ----------
    # 1) Trade-off scatter
    plt.figure()
    for m in modes:
        sub = df_succ[df_succ["mode"] == m]
        if len(sub) == 0:
            continue
        plt.scatter(sub["capacity_bits"], sub["delta_ppl"], label=m, alpha=0.7)
    plt.xlabel("Capacity (bits)")
    plt.ylabel("ΔPPL (stego - orig)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_tradeoff.png", dpi=200)
    plt.close()

    # 2) Bar: mean capacity
    plt.figure()
    cap_plot = cap_stats.set_index("mode").reindex(modes)["mean"]
    cap_plot.plot(kind="bar")
    plt.ylabel("Mean capacity (bits)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_capacity_bar.png", dpi=200)
    plt.close()

    # 3) Bar: success rate
    plt.figure()
    succ_plot = succ.set_index("mode").reindex(modes)["success_rate_pct"]
    succ_plot.plot(kind="bar")
    plt.ylabel("Success rate (%)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_success_bar.png", dpi=200)
    plt.close()

    # 4) Bar: mean ΔPPL (success only)
    plt.figure()
    delta_plot = ppl_stats.set_index("mode").reindex([m for m in modes if m in ppl_stats["mode"].values])["delta_ppl_mean"]
    delta_plot.plot(kind="bar")
    plt.ylabel("Mean ΔPPL (success only)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_delta_ppl_bar.png", dpi=200)
    plt.close()

    # ---------- LaTeX snippets ----------
    latex_snippets = []
    latex_snippets.append("% --- Table: capacity ---")
    latex_snippets.append(latex_capacity_table(cap_stats))
    latex_snippets.append("\n% --- Table: success ---")
    latex_snippets.append(latex_success_table(succ))
    latex_snippets.append("\n% --- Table: perplexity (success-only) ---")
    latex_snippets.append(latex_ppl_table(ppl_stats))
    latex_snippets_text = "\n".join(latex_snippets)
    (outdir / "latex_table_rows.txt").write_text(latex_snippets_text, encoding="utf-8")

    # ---------- Write a short stats summary ----------
    summary_lines = []
    summary_lines.append("=== Capacity: Kruskal–Wallis ===")
    summary_lines.append(f"stat={kw_stat:.4f}, p={kw_p:.6g}")
    summary_lines.append("")
    summary_lines.append("=== Pairwise capacity: Mann–Whitney U (Holm-corrected) ===")
    summary_lines.append(pairwise_capacity.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("=== Success: Chi-square on success/fail by mode ===")
    summary_lines.append(f"chi2={chi2_stat:.4f}, dof={chi2_dof}, p={chi2_p:.6g}")
    summary_lines.append("")
    summary_lines.append("=== Pairwise success: Fisher exact (Holm-corrected) ===")
    summary_lines.append(pairwise_success.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("=== Correlations (success only): capacity vs ΔPPL ===")
    summary_lines.append(corr_df.to_string(index=False))

    (outdir / "stats_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Done. Wrote outputs to: {outdir.resolve()}")
    print(f"- Tables: {outdir/'table_capacity_stats.csv'}, {outdir/'table_success_rates.csv'}, {outdir/'table_ppl_stats_success_only.csv'}")
    print(f"- Pairwise tests: {outdir/'pairwise_capacity_mannwhitney_holm.csv'}, {outdir/'pairwise_success_fisher_holm.csv'}")
    print(f"- Correlations: {outdir/'correlations_capacity_vs_delta_ppl.csv'}")
    print(f"- Figures: {outdir/'fig_tradeoff.png'}, {outdir/'fig_capacity_bar.png'}, {outdir/'fig_success_bar.png'}, {outdir/'fig_delta_ppl_bar.png'}")
    print(f"- LaTeX rows: {outdir/'latex_table_rows.txt'}")
    print(f"- Summary: {outdir/'stats_summary.txt'}")


if __name__ == "__main__":
    main()