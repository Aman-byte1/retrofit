"""
Analysis and visualization for architecture comparison.

Generates:
- Training loss curves (all 4 architectures overlaid)
- Validation perplexity comparison table
- Throughput vs sequence length scaling plot
- VRAM vs sequence length scaling plot
- Pareto frontier: quality vs efficiency
- Parameter efficiency scatter plot
- Comprehensive LaTeX tables
- Full markdown research report
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("analyze")

ARCH_COLORS = {
    "transformer": "#e74c3c",
    "hrm": "#3498db",
    "mamba": "#2ecc71",
    "hybrid": "#9b59b6",
}

ARCH_LABELS = {
    "transformer": "Transformer (RoPE+SwiGLU)",
    "hrm": "HRM (Dual-Timescale)",
    "mamba": "Mamba (Selective SSM)",
    "hybrid": "Hybrid (Mamba+Attn)",
}

ARCH_MARKERS = {
    "transformer": "o",
    "hrm": "s",
    "mamba": "D",
    "hybrid": "^",
}


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load all results from each architecture subdirectory (canonical or timestamped)."""
    all_results = {}
    
    for arch in ["transformer", "hrm", "mamba", "hybrid"]:
        candidate_dirs = [os.path.join(results_dir, arch)]
        if os.path.exists(results_dir):
            for d in sorted(os.listdir(results_dir)):
                if d.startswith(f"{arch}_") and os.path.isdir(os.path.join(results_dir, d)):
                    candidate_dirs.append(os.path.join(results_dir, d))
        
        arch_data = {"arch": arch}
        
        for arch_dir in candidate_dirs:
            if not os.path.exists(arch_dir):
                continue
            
            # Training summary
            for fname in ["summary.json", "results.json"]:
                fpath = os.path.join(arch_dir, fname)
                if os.path.exists(fpath) and "training" not in arch_data:
                    with open(fpath, encoding="utf-8") as f:
                        arch_data["training"] = json.load(f)
            
            # Benchmark results
            bpath = os.path.join(arch_dir, "benchmark.json")
            if os.path.exists(bpath) and "benchmark" not in arch_data:
                with open(bpath, encoding="utf-8") as f:
                    arch_data["benchmark"] = json.load(f)
            
            # Step logs (.json or .jsonl)
            if "step_logs" not in arch_data:
                jsonl_path = os.path.join(arch_dir, "step_logs.jsonl")
                json_path = os.path.join(arch_dir, "step_logs.json")
                if os.path.exists(jsonl_path):
                    logs = []
                    with open(jsonl_path, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                logs.append(json.loads(line))
                    arch_data["step_logs"] = logs
                elif os.path.exists(json_path):
                    with open(json_path, encoding="utf-8") as f:
                        arch_data["step_logs"] = json.load(f)
        
        if len(arch_data) > 1:
            all_results[arch] = arch_data
    
    logger.info(f"Loaded results for: {list(all_results.keys())}")
    return all_results


def plot_loss_curves(all_results: Dict, output_dir: str):
    """Plot training loss curves for all architectures."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for arch, data in all_results.items():
        if "step_logs" not in data:
            continue
        
        steps = [s["global_step"] for s in data["step_logs"]]
        losses = [s["loss"] for s in data["step_logs"]]
        
        # Smooth loss for plotting
        window = max(1, len(losses) // 100)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
            plot_steps = steps[:len(smoothed)]
        else:
            smoothed = losses
            plot_steps = steps
        
        label = ARCH_LABELS.get(arch, arch)
        color = ARCH_COLORS.get(arch, "#999")
        
        ax1.plot(plot_steps, smoothed, color=color, label=label, linewidth=2, alpha=0.9)
    
    ax1.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Training Loss Convergence", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Epoch-level val PPL
    for arch, data in all_results.items():
        if "training" not in data:
            continue
        
        epochs = [e["epoch"] + 1 for e in data["training"]["epoch_logs"]]
        val_ppls = [e["val_perplexity"] for e in data["training"]["epoch_logs"]]
        
        label = ARCH_LABELS.get(arch, arch)
        color = ARCH_COLORS.get(arch, "#999")
        marker = ARCH_MARKERS.get(arch, "o")
        
        ax2.plot(epochs, val_ppls, color=color, label=label, linewidth=2,
                marker=marker, markersize=8, alpha=0.9)
    
    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Validation Perplexity ↓", fontsize=12, fontweight="bold")
    ax2.set_title("Validation Perplexity (Lower = Better)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved loss curves to {save_path}")


def plot_throughput_scaling(all_results: Dict, output_dir: str):
    """Plot throughput vs sequence length for all architectures."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for arch, data in all_results.items():
        if "benchmark" not in data:
            continue
        
        bench = data["benchmark"]
        label = ARCH_LABELS.get(arch, arch)
        color = ARCH_COLORS.get(arch, "#999")
        marker = ARCH_MARKERS.get(arch, "o")
        
        # Batch throughput
        for key in ["batch_throughput", "single_throughput"]:
            if key not in bench:
                continue
            
            results = [r for r in bench[key] if r.get("status") == "success"]
            if not results:
                continue
            
            seq_lens = [r["seq_len"] for r in results]
            tps = [r["tokens_per_sec"] for r in results]
            vram = [r["peak_vram_mb"] for r in results]
            
            ax_target = ax1 if key == "batch_throughput" else ax1
            ax_target.plot(seq_lens, tps, color=color, label=label if key == "batch_throughput" else None,
                          linewidth=2, marker=marker, markersize=8, alpha=0.9)
            
            ax2.plot(seq_lens, vram, color=color, label=label if key == "batch_throughput" else None,
                    linewidth=2, marker=marker, markersize=8, alpha=0.9,
                    linestyle="--" if key == "single_throughput" else "-")
    
    ax1.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Throughput (tokens/sec) ↑", fontsize=12, fontweight="bold")
    ax1.set_title("Inference Throughput Scaling", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)
    
    ax2.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Peak VRAM (MB) ↓", fontsize=12, fontweight="bold")
    ax2.set_title("Memory Scaling (Peak VRAM)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "throughput_scaling.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved throughput scaling to {save_path}")


def plot_pareto(all_results: Dict, output_dir: str):
    """Plot Pareto frontier: quality (PPL) vs efficiency (throughput)."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for arch, data in all_results.items():
        if "training" not in data or "benchmark" not in data:
            continue
        
        ppl = data["training"]["best_val_perplexity"]
        
        # Get throughput at seq_len=512
        bench = data["benchmark"].get("batch_throughput", [])
        tps_512 = 0
        for r in bench:
            if r.get("seq_len") == 512 and r.get("status") == "success":
                tps_512 = r["tokens_per_sec"]
                break
        
        if tps_512 == 0:
            continue
        
        params = data["training"]["metadata"]["total_params"]
        label = ARCH_LABELS.get(arch, arch)
        color = ARCH_COLORS.get(arch, "#999")
        marker = ARCH_MARKERS.get(arch, "o")
        
        ax.scatter(tps_512, ppl, s=300, c=color, marker=marker,
                  edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(
            f"{label}\n({params/1e6:.1f}M params)",
            (tps_512, ppl),
            textcoords="offset points",
            xytext=(15, 10),
            fontsize=10,
            fontweight="bold",
        )
    
    ax.set_xlabel("Throughput at seq_len=512 (tokens/sec) →", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Perplexity ↓", fontsize=12, fontweight="bold")
    ax.set_title("Quality-Efficiency Pareto Frontier\n(Lower-Right = Better)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "pareto_frontier.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Pareto frontier to {save_path}")


def generate_latex_table(all_results: Dict, output_dir: str):
    """Generate LaTeX results table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Architecture Comparison on Amharic Wikipedia Language Modeling}",
        r"\label{tab:arch_comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Architecture} & \textbf{Params} & \textbf{Val PPL $\downarrow$} & \textbf{Train Time} & \textbf{Tok/s $\uparrow$} & \textbf{VRAM} & \textbf{Complexity} \\",
        r"\midrule",
    ]
    
    for arch in ["transformer", "hrm", "mamba", "hybrid"]:
        if arch not in all_results:
            continue
        
        data = all_results[arch]
        label = ARCH_LABELS.get(arch, arch)
        
        params = data.get("training", {}).get("metadata", {}).get("total_params", 0)
        ppl = data.get("training", {}).get("best_val_perplexity", 0)
        train_time = data.get("training", {}).get("total_train_time_min", 0)
        
        # Get throughput at seq_len=512
        tps = 0
        vram = 0
        for r in data.get("benchmark", {}).get("batch_throughput", []):
            if r.get("seq_len") == 512 and r.get("status") == "success":
                tps = r["tokens_per_sec"]
                vram = r["peak_vram_mb"]
                break
        
        complexity = r"$O(N^2)$" if arch == "transformer" else r"$O(N)$"
        if arch == "hybrid":
            complexity = r"$O(N \cdot K)$"
        
        lines.append(
            f"{label} & {params/1e6:.1f}M & {ppl:.1f} & {train_time:.0f}min & "
            f"{tps:.0f} & {vram:.0f}MB & {complexity} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    table = "\n".join(lines)
    save_path = os.path.join(output_dir, "results_table.tex")
    Path(save_path).write_text(table)
    logger.info(f"Saved LaTeX table to {save_path}")
    print("\n" + table)


def generate_report(all_results: Dict, output_dir: str):
    """Generate comprehensive markdown report."""
    report = [
        "# 🏛️ Architecture Comparison for Amharic Language Modeling\n",
        "**Task**: Causal language modeling on Amharic Wikipedia  ",
        "**Tokenizer**: RL-trained Amharic subword tokenizer (3,919 tokens)  ",
        "**GPU**: NVIDIA A100 (40GB)  ",
        "**Precision**: bfloat16 mixed-precision  \n",
        "---\n",
        "## 📊 Summary Results\n",
        "| Architecture | Parameters | Val PPL ↓ | Train Time | Throughput (tok/s) ↑ | Peak VRAM | Complexity |",
        "|---|---|---|---|---|---|---|",
    ]
    
    for arch in ["transformer", "hrm", "mamba", "hybrid"]:
        if arch not in all_results:
            continue
        
        data = all_results[arch]
        label = ARCH_LABELS.get(arch, arch)
        params = data.get("training", {}).get("metadata", {}).get("total_params", 0)
        ppl = data.get("training", {}).get("best_val_perplexity", 0)
        train_time = data.get("training", {}).get("total_train_time_min", 0)
        
        tps = 0
        vram = 0
        for r in data.get("benchmark", {}).get("batch_throughput", []):
            if r.get("seq_len") == 512 and r.get("status") == "success":
                tps = r["tokens_per_sec"]
                vram = r["peak_vram_mb"]
                break
        
        complexity_map = {
            "transformer": "O(N²)",
            "hrm": "O(N)",
            "mamba": "O(N)",
            "hybrid": "O(N·K)",
        }
        
        report.append(
            f"| **{label}** | {params/1e6:.1f}M | {ppl:.1f} | {train_time:.0f} min | "
            f"{tps:,.0f} | {vram:.0f} MB | {complexity_map.get(arch, '?')} |"
        )
    
    # Per-architecture details
    report.append("\n---\n")
    
    for arch in ["transformer", "hrm", "mamba", "hybrid"]:
        if arch not in all_results:
            continue
        
        data = all_results[arch]
        label = ARCH_LABELS.get(arch, arch)
        
        report.append(f"## {label}\n")
        
        if "training" in data:
            t = data["training"]
            report.append(f"- **Parameters**: {t['metadata']['total_params']:,}")
            report.append(f"- **Best Val PPL**: {t['best_val_perplexity']:.2f}")
            report.append(f"- **Training Time**: {t['total_train_time_min']:.1f} min")
            report.append(f"- **d_model**: {t['metadata']['d_model']}")
            
            report.append("\n### Epoch-by-Epoch\n")
            report.append("| Epoch | Train Loss | Train PPL | Val Loss | Val PPL | Tok/s |")
            report.append("|---|---|---|---|---|---|")
            for e in t["epoch_logs"]:
                report.append(
                    f"| {e['epoch']+1} | {e['avg_loss']:.4f} | {e['avg_perplexity']:.1f} | "
                    f"{e['val_loss']:.4f} | {e['val_perplexity']:.1f} | {e['avg_tokens_per_sec']:,.0f} |"
                )
        
        if "benchmark" in data:
            b = data["benchmark"]
            report.append("\n### Throughput Scaling\n")
            report.append("| Seq Len | Throughput (tok/s) | Peak VRAM (MB) |")
            report.append("|---|---|---|")
            for r in b.get("batch_throughput", []):
                if r.get("status") == "success":
                    report.append(f"| {r['seq_len']} | {r['tokens_per_sec']:,.0f} | {r['peak_vram_mb']:.0f} |")
            
            if "generation_latency" in b and b["generation_latency"].get("status") == "success":
                gl = b["generation_latency"]
                report.append(f"\n### Generation Latency\n")
                report.append(f"- **First token**: {gl['avg_first_token_ms']:.1f}ms")
                report.append(f"- **Per token (avg)**: {gl['avg_per_token_ms']:.1f}ms")
                report.append(f"- **Per token (p95)**: {gl['p95_per_token_ms']:.1f}ms")
                report.append(f"- **Generation speed**: {gl['gen_tokens_per_sec']:.1f} tok/s")
        
        report.append("\n---\n")
    
    report_str = "\n".join(report)
    save_path = os.path.join(output_dir, "report.md")
    Path(save_path).write_text(report_str)
    logger.info(f"Saved report to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/analysis")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = load_results(args.results_dir)
    
    if not all_results:
        logger.error("No results found!")
        return
    
    # Generate report and tables
    generate_report(all_results, args.output_dir)
    generate_latex_table(all_results, args.output_dir)
    
    # Generate plots
    if not args.no_plots:
        try:
            plot_loss_curves(all_results, args.output_dir)
            plot_throughput_scaling(all_results, args.output_dir)
            plot_pareto(all_results, args.output_dir)
        except Exception as e:
            logger.warning(f"Plotting error: {e}")
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
