"""
Analysis and plotting for Retrofit experiments.

Generates:
- Efficiency comparison plots (quality vs compute)
- Layer importance heatmaps
- Per-language breakdown charts
- Ablation study tables

Usage:
    python -m retrofit.analyze --results-dir experiments/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("retrofit.analyze")


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all evaluation results from experiment subdirectories."""
    all_records = []
    
    for json_file in sorted(results_dir.rglob("eval_results_*.json")):
        method = json_file.stem.replace("eval_results_", "")
        experiment_dir = json_file.parent.name
        
        with open(json_file) as f:
            results = json.load(f)
        
        for lang, stats in results.items():
            all_records.append({
                "experiment": experiment_dir,
                "method": method,
                "language": lang,
                **stats,
            })
    
    if not all_records:
        logger.warning(f"No results found in {results_dir}")
        return pd.DataFrame()
    
    return pd.DataFrame(all_records)


def load_all_detailed(results_dir: Path) -> pd.DataFrame:
    """Load all per-sample detailed results."""
    dfs = []
    
    for csv_file in sorted(results_dir.rglob("eval_detailed_*.csv")):
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def plot_efficiency_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Plot: Combined Score vs Method (bar chart with error bars).
    
    This is the key figure for the research paper — shows quality
    achieved by each method.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [
        ("avg_combined", "Combined Score ↑"),
        ("avg_cer", "CER ↓"),
        ("avg_speaker_sim", "Speaker Similarity ↑"),
    ]
    
    method_order = ["zero_shot", "full_finetune", "uniform_lora", "targeted_lora"]
    method_labels = {
        "zero_shot": "Zero-Shot",
        "full_finetune": "Full Fine-Tune",
        "uniform_lora": "Uniform LoRA",
        "targeted_lora": "Targeted LoRA",
    }
    colors = {
        "zero_shot": "#95a5a6",
        "full_finetune": "#e74c3c",
        "uniform_lora": "#3498db",
        "targeted_lora": "#2ecc71",
    }
    
    for ax, (metric, label) in zip(axes, metrics):
        methods_present = [m for m in method_order if m in df["method"].unique()]
        
        for i, method in enumerate(methods_present):
            method_data = df[df["method"] == method]
            avg = method_data[metric].mean()
            std = method_data[metric].std() if len(method_data) > 1 else 0
            
            ax.bar(
                i, avg,
                yerr=std,
                color=colors.get(method, "#999"),
                label=method_labels.get(method, method),
                capsize=5,
                edgecolor="white",
                linewidth=1.5,
            )
        
        ax.set_xlabel("Method")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(range(len(methods_present)))
        ax.set_xticklabels(
            [method_labels.get(m, m) for m in methods_present],
            rotation=30,
            ha="right",
        )
    
    plt.suptitle("Retrofit: Efficiency Comparison Across Methods", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    save_path = output_dir / "efficiency_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved efficiency comparison plot to {save_path}")


def plot_language_breakdown(df: pd.DataFrame, output_dir: Path):
    """Plot per-language performance for each method."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot = df.pivot_table(
        values="avg_combined",
        index="method",
        columns="language",
        aggfunc="mean",
    )
    
    if pivot.empty:
        logger.warning("No data to plot for language breakdown")
        return
    
    pivot.plot(kind="bar", ax=ax, width=0.8, edgecolor="white", linewidth=1.5)
    
    ax.set_ylabel("Combined Score")
    ax.set_title("Performance by Language and Method", fontsize=14, fontweight="bold")
    ax.legend(title="Language", loc="upper right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    
    plt.tight_layout()
    save_path = output_dir / "language_breakdown.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved language breakdown plot to {save_path}")


def plot_parameter_efficiency(output_dir: Path):
    """
    Plot: Performance vs Trainable Parameters.
    
    Shows the Pareto frontier of quality vs model efficiency.
    """
    import matplotlib.pyplot as plt
    
    # Load parameter counts from experiment configs
    param_data = []
    
    for config_file in sorted(output_dir.rglob("config.yaml")):
        import yaml
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        
        results_file = config_file.parent / f"eval_results_{cfg.get('adaptation_method', 'unknown')}.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            
            avg_combined = np.mean([s["avg_combined"] for s in results.values()])
            
            # Estimate parameter count
            method = cfg.get("adaptation_method", "unknown")
            rank = cfg.get("lora", {}).get("rank", 8)
            
            # Rough estimates (will be replaced by actual counts from training logs)
            param_estimates = {
                "zero_shot": 0,
                "uniform_lora": rank * 2 * 1024 * 22,  # Approximate
                "targeted_lora": rank * 2 * 1024 * 8,
                "full_finetune": 300_000_000,
            }
            
            param_data.append({
                "method": method,
                "trainable_params": param_estimates.get(method, 0),
                "combined_score": avg_combined,
            })
    
    if not param_data:
        logger.warning("No data for parameter efficiency plot")
        return
    
    df = pd.DataFrame(param_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "zero_shot": "#95a5a6",
        "full_finetune": "#e74c3c",
        "uniform_lora": "#3498db",
        "targeted_lora": "#2ecc71",
    }
    
    for _, row in df.iterrows():
        ax.scatter(
            row["trainable_params"],
            row["combined_score"],
            s=200,
            c=colors.get(row["method"], "#999"),
            edgecolors="black",
            linewidth=1.5,
            zorder=5,
        )
        ax.annotate(
            row["method"].replace("_", " ").title(),
            (row["trainable_params"], row["combined_score"]),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10,
        )
    
    ax.set_xlabel("Trainable Parameters", fontsize=12)
    ax.set_ylabel("Combined Score ↑", fontsize=12)
    ax.set_title("Parameter Efficiency: Quality vs Trainable Parameters", fontsize=14, fontweight="bold")
    ax.set_xscale("symlog")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / "parameter_efficiency.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved parameter efficiency plot to {save_path}")


def generate_latex_table(df: pd.DataFrame, output_dir: Path):
    """Generate a LaTeX-formatted results table for the research paper."""
    
    method_order = ["zero_shot", "full_finetune", "uniform_lora", "targeted_lora"]
    method_labels = {
        "zero_shot": "Zero-Shot (baseline)",
        "full_finetune": "Full Fine-Tune",
        "uniform_lora": "Uniform LoRA",
        "targeted_lora": "Targeted LoRA (ours)",
    }
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Voice Cloning Quality: CER, Speaker Similarity, and Combined Score}",
        r"\label{tab:results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{CER $\downarrow$} & \textbf{Speaker Sim $\uparrow$} & \textbf{Combined $\uparrow$} \\",
        r"\midrule",
    ]
    
    for method in method_order:
        method_data = df[df["method"] == method]
        if method_data.empty:
            continue
        
        avg_cer = method_data["avg_cer"].mean()
        avg_sim = method_data["avg_speaker_sim"].mean()
        avg_comb = method_data["avg_combined"].mean()
        
        label = method_labels.get(method, method)
        lines.append(f"{label} & {avg_cer:.3f} & {avg_sim:.3f} & {avg_comb:.3f} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    table_str = "\n".join(lines)
    
    save_path = output_dir / "results_table.tex"
    save_path.write_text(table_str)
    logger.info(f"Saved LaTeX table to {save_path}")
    
    print("\n" + table_str)


def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """Generate a markdown summary report."""
    
    report = ["# Retrofit: Experiment Results\n"]
    report.append(f"Total experiments: {df['experiment'].nunique()}")
    report.append(f"Methods evaluated: {', '.join(df['method'].unique())}")
    report.append(f"Languages: {', '.join(df['language'].unique())}\n")
    
    report.append("## Results by Method\n")
    report.append("| Method | CER ↓ | Speaker Sim ↑ | Combined ↑ | N |")
    report.append("|--------|-------|---------------|------------|---|")
    
    for method in df["method"].unique():
        method_data = df[df["method"] == method]
        report.append(
            f"| {method} | "
            f"{method_data['avg_cer'].mean():.3f} | "
            f"{method_data['avg_speaker_sim'].mean():.3f} | "
            f"{method_data['avg_combined'].mean():.3f} | "
            f"{method_data['count'].sum()} |"
        )
    
    report.append("\n## Results by Language\n")
    report.append("| Language | Method | CER ↓ | Speaker Sim ↑ | Combined ↑ |")
    report.append("|----------|--------|-------|---------------|------------|")
    
    for _, row in df.iterrows():
        report.append(
            f"| {row['language']} | {row['method']} | "
            f"{row['avg_cer']:.3f} | {row['avg_speaker_sim']:.3f} | "
            f"{row['avg_combined']:.3f} |"
        )
    
    report_str = "\n".join(report)
    
    save_path = output_dir / "results_report.md"
    save_path.write_text(report_str)
    logger.info(f"Saved summary report to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Retrofit: Analyze Experiment Results")
    parser.add_argument("--results-dir", type=str, default="experiments",
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Where to save plots (defaults to results-dir)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir or args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    df = load_all_results(results_dir)
    
    if df.empty:
        logger.error("No results found! Run experiments first.")
        return
    
    logger.info(f"Loaded {len(df)} result records from {results_dir}")
    
    # Generate report
    generate_summary_report(df, output_dir)
    
    # Generate LaTeX table
    generate_latex_table(df, output_dir)
    
    # Generate plots
    if not args.no_plots:
        try:
            plot_efficiency_comparison(df, output_dir)
            plot_language_breakdown(df, output_dir)
            plot_parameter_efficiency(output_dir)
        except ImportError as e:
            logger.warning(f"Plotting failed (missing dependency): {e}")
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
