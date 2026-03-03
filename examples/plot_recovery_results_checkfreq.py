#!/usr/bin/env python
# coding=utf-8
"""
Plot recovery benchmark results: CheckFreq vs Standard PyTorch resume.
Usage: python plot_recovery_results_checkfreq.py <result_dir>
"""
import sys
import os
import csv
import numpy as np

def read_csv_results(csv_path):
    """Read timing results from CSV file."""
    results = []
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return results
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def extract_metric(results, key, default=0.0):
    """Extract a metric from results list, converting to float."""
    values = []
    for r in results:
        if key in r and r[key] not in (None, ''):
            try:
                values.append(float(r[key]))
            except (ValueError, TypeError):
                pass
    return values


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_recovery_results_checkfreq.py <result_dir>")
        sys.exit(1)

    result_dir = sys.argv[1]
    checkfreq_csv = os.path.join(result_dir, "checkfreq_results.csv")
    standard_csv = os.path.join(result_dir, "standard_results.csv")

    checkfreq_results = read_csv_results(checkfreq_csv)
    standard_results = read_csv_results(standard_csv)

    if not checkfreq_results and not standard_results:
        print("No results found to plot.")
        sys.exit(1)

    # Extract key metrics
    metrics_to_compare = [
        ("first_loop_after_resume", "First Loop After Resume (s)"),
        ("checkpoint_resume", "Checkpoint Resume (s)"),
        ("checkfreq_restore", "CheckFreq Restore (s)"),
        ("total_execution_time", "Total Execution Time (s)"),
        ("model_loading", "Model Loading (s)"),
        ("accelerator_preparation", "Accelerator Preparation (s)"),
    ]

    print("=" * 70)
    print("Recovery Benchmark Results: CheckFreq vs Standard")
    print("=" * 70)

    summary_lines = []
    for key, label in metrics_to_compare:
        cf_vals = extract_metric(checkfreq_results, key)
        std_vals = extract_metric(standard_results, key)

        line = f"\n{label}:"
        if cf_vals:
            line += f"\n  CheckFreq:  mean={np.mean(cf_vals):.2f}s, std={np.std(cf_vals):.2f}s, n={len(cf_vals)}"
        if std_vals:
            line += f"\n  Standard:   mean={np.mean(std_vals):.2f}s, std={np.std(std_vals):.2f}s, n={len(std_vals)}"
        if cf_vals and std_vals:
            speedup = np.mean(std_vals) / np.mean(cf_vals) if np.mean(cf_vals) > 0 else float('inf')
            diff = np.mean(std_vals) - np.mean(cf_vals)
            line += f"\n  Speedup:    {speedup:.2f}x (diff={diff:.2f}s)"
        print(line)
        summary_lines.append(line)

    # Try to generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Bar chart: first_loop_after_resume comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: First loop after resume
        cf_first_loop = extract_metric(checkfreq_results, "first_loop_after_resume")
        std_first_loop = extract_metric(standard_results, "first_loop_after_resume")

        methods = []
        means = []
        stds = []
        colors = []

        if cf_first_loop:
            methods.append("CheckFreq")
            means.append(np.mean(cf_first_loop))
            stds.append(np.std(cf_first_loop))
            colors.append('#2196F3')
        if std_first_loop:
            methods.append("Standard")
            means.append(np.mean(std_first_loop))
            stds.append(np.std(std_first_loop))
            colors.append('#FF9800')

        if methods:
            bars = axes[0].bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
            axes[0].set_ylabel("Time (seconds)")
            axes[0].set_title("First Loop After Resume")
            for bar, mean in zip(bars, means):
                axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                           f'{mean:.2f}s', ha='center', va='bottom', fontweight='bold')

        # Plot 2: Breakdown of resume stages
        breakdown_keys = [
            ("model_loading", "Model Loading"),
            ("checkfreq_restore", "CF Restore"),
            ("checkpoint_resume", "Chkpt Resume"),
            ("accelerator_preparation", "Accelerator Prep"),
        ]

        x = np.arange(len(breakdown_keys))
        width = 0.35

        cf_means = []
        std_means = []
        labels = []
        for key, label in breakdown_keys:
            cf_v = extract_metric(checkfreq_results, key)
            std_v = extract_metric(standard_results, key)
            cf_means.append(np.mean(cf_v) if cf_v else 0)
            std_means.append(np.mean(std_v) if std_v else 0)
            labels.append(label)

        axes[1].bar(x - width / 2, cf_means, width, label='CheckFreq', color='#2196F3', alpha=0.8)
        axes[1].bar(x + width / 2, std_means, width, label='Standard', color='#FF9800', alpha=0.8)
        axes[1].set_ylabel("Time (seconds)")
        axes[1].set_title("Recovery Time Breakdown")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=15, ha='right')
        axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(result_dir, "checkfreq_recovery_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")

    except ImportError:
        print("\nmatplotlib not available. Skipping plot generation.")

    # Save text summary
    summary_path = os.path.join(result_dir, "recovery_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Recovery Benchmark Results: CheckFreq vs Standard\n")
        f.write("=" * 70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
