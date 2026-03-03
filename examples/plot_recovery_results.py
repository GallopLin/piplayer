#!/usr/bin/env python3
"""
Plot recovery benchmark results: Standard vs PipeLayer vs CheckFreq.

Main comparison metric: restore time + first training step time.

Usage:
  python plot_recovery_results.py [output_dir]

Produces:
  1. recovery_restore_first_step.png  - Stacked bar: restore + first step
  2. recovery_breakdown.png           - Horizontal stacked bar: full stage breakdown
  3. checkpoint_load_time.png         - Pure checkpoint loading time comparison
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

HOME = os.path.expanduser("~")

DEFAULT_PIPELAYER_CSV = f"{HOME}/download/ckpt/training_time_log.csv"
DEFAULT_STANDARD_CSV = f"{HOME}/download/ckpt_standard_resume/training_time_log.csv"
DEFAULT_CHECKFREQ_CSV = (
    f"{HOME}/download/pipelayer/examples/checkfreq_experiment/"
    "output_checkfreq_restore/training_time_log.csv"
)

# ── Colour palette ──────────────────────────────────────────────
C_STANDARD  = '#E27733'   # orange
C_PIPELAYER = '#9B59B6'   # purple
C_CHECKFREQ = '#2980B9'   # blue


# ================================================================
#  Data loading
# ================================================================
def load_data(result_dir=None):
    """Return (pipelayer_df, standard_df, checkfreq_df)."""
    if result_dir and os.path.isdir(result_dir):
        pl_csv = os.path.join(result_dir, "pipelayer_results.csv")
        st_csv = os.path.join(result_dir, "standard_results.csv")
        cf_csv = os.path.join(result_dir, "checkfreq_results.csv")
        if not os.path.exists(pl_csv):
            pl_csv = DEFAULT_PIPELAYER_CSV
        if not os.path.exists(st_csv):
            st_csv = DEFAULT_STANDARD_CSV
        if not os.path.exists(cf_csv):
            cf_csv = DEFAULT_CHECKFREQ_CSV
    else:
        pl_csv = DEFAULT_PIPELAYER_CSV
        st_csv = DEFAULT_STANDARD_CSV
        cf_csv = DEFAULT_CHECKFREQ_CSV

    pl_df = pd.read_csv(pl_csv) if os.path.exists(pl_csv) else None
    st_df = pd.read_csv(st_csv) if os.path.exists(st_csv) else None
    cf_df = pd.read_csv(cf_csv) if os.path.exists(cf_csv) else None

    return pl_df, st_df, cf_df


# ================================================================
#  Helper: extract restore-time & first-step per method
# ================================================================
def _get_restore_and_first_step(df, method):
    """Return (restore_mean, restore_std, step_mean, step_std) for a method."""
    if df is None:
        return None

    if method == 'standard':
        restore_col = 'checkpoint_resume'
    elif method == 'pipelayer':
        restore_col = 'pipelayer_setup'
    elif method == 'checkfreq':
        restore_col = 'checkfreq_restore'
    else:
        return None

    if restore_col not in df.columns or 'first_training_step' not in df.columns:
        return None

    r = df[restore_col].dropna()
    s = df['first_training_step'].dropna()
    return (
        r.mean(), r.std() if len(r) > 1 else 0,
        s.mean(), s.std() if len(s) > 1 else 0,
    )


# ================================================================
#  Figure 0 – first_loop_after_resume bar chart (like old chart)
# ================================================================
def plot_first_loop_comparison(pl_df, st_df, cf_df, output_dir):
    """Bar chart: first_loop_after_resume comparison (3 methods)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    means = []
    stds = []
    colors = []

    for label, df, color in [
        ('Standard\nPyTorch',        st_df, C_STANDARD),
        ('CheckFreq',                cf_df, C_CHECKFREQ),
        ('PipeLayer\n(MultiStream)', pl_df, C_PIPELAYER),
    ]:
        if df is not None and 'first_loop_after_resume' in df.columns:
            vals = df['first_loop_after_resume'].dropna()
            methods.append(label)
            means.append(vals.mean())
            stds.append(vals.std() if len(vals) > 1 else 0)
            colors.append(color)

    if not methods:
        print("No data for first_loop bar chart")
        return

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, width=0.5, color=colors,
                  edgecolor='black', linewidth=1.2, capsize=8,
                  error_kw={'linewidth': 2})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{mean:.1f}s', ha='center', va='bottom',
                fontsize=20, fontweight='bold')

    # Speedup annotations: compare every bar against the slowest
    max_mean = max(means)
    max_idx = means.index(max_mean)
    offsets_used = 0
    for i, m in enumerate(means):
        if i != max_idx and m > 0:
            speedup = max_mean / m
            x_txt = x[i] + 0.35
            y_txt = max_mean * (0.65 - 0.12 * offsets_used)
            ax.annotate(f'{speedup:.2f}x faster',
                        xy=(x[i], m),
                        xytext=(x_txt, y_txt),
                        fontsize=16, color=colors[i], fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
            offsets_used += 1

    ax.set_ylabel('Time to First Training Step (s)', fontsize=22)
    ax.set_title('OPT-1.3B Checkpoint Recovery Latency', fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=20)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, max(means) * 1.3 if means else 70)
    ax.grid(axis='y', alpha=0.3)

    run_counts = []
    for label, df in [('Standard', st_df), ('CheckFreq', cf_df), ('PipeLayer', pl_df)]:
        if df is not None:
            run_counts.append(f'{label} n={len(df)}')
    ax.text(0.02, 0.98, ', '.join(run_counts),
            transform=ax.transAxes, fontsize=14, va='top', color='grey')

    plt.tight_layout()
    path = os.path.join(output_dir, 'recovery_bar_chart.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ================================================================
#  Figure 1 – Restore + First Training Step (stacked bar)
# ================================================================
def plot_restore_first_step(pl_df, st_df, cf_df, output_dir):
    """Stacked bar chart: restore time (bottom) + first training step (top)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    entries = []  # (label, restore_mean, restore_std, step_mean, step_std, color)
    for label, df, method, color in [
        ('Standard\nPyTorch',        st_df, 'standard',  C_STANDARD),
        ('CheckFreq',                cf_df, 'checkfreq', C_CHECKFREQ),
        ('PipeLayer\n(MultiStream)', pl_df, 'pipelayer', C_PIPELAYER),
    ]:
        vals = _get_restore_and_first_step(df, method)
        if vals is not None:
            entries.append((label, *vals, color))

    if not entries:
        print("No data for restore+first-step chart")
        return

    labels   = [e[0] for e in entries]
    r_means  = [e[1] for e in entries]
    s_means  = [e[3] for e in entries]
    colors   = [e[5] for e in entries]
    totals   = [r + s for r, s in zip(r_means, s_means)]

    x = np.arange(len(labels))
    width = 0.50

    # Bottom part: restore time
    bars_r = ax.bar(x, r_means, width, label='Restore / Load Time',
                    color=colors, edgecolor='black', linewidth=1.0, alpha=0.85)
    # Top part: first training step
    bars_s = ax.bar(x, s_means, width, bottom=r_means,
                    label='First Training Step',
                    color=colors, edgecolor='black', linewidth=1.0, alpha=0.50,
                    hatch='///')

    # Value labels inside bars
    for i, (rm, sm) in enumerate(zip(r_means, s_means)):
        if rm > 1.5:
            ax.text(x[i], rm / 2, f'{rm:.2f}s', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
        ax.text(x[i], rm + sm / 2, f'{sm:.2f}s', ha='center', va='center',
                fontsize=13, fontweight='bold', color='black')
        # Total on top
        ax.text(x[i], rm + sm + 0.6, f'{rm + sm:.2f}s',
                ha='center', va='bottom', fontsize=18, fontweight='bold')

    # Speedup annotations relative to the slowest
    max_total = max(totals)
    for i, t in enumerate(totals):
        if t < max_total and t > 0:
            speedup = max_total / t
            ax.annotate(f'{speedup:.1f}× faster',
                        xy=(x[i], totals[i] + 0.3),
                        xytext=(x[i] + 0.35, max_total * 0.75),
                        fontsize=15, color=colors[i], fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))

    ax.set_ylabel('Time (seconds)', fontsize=20)
    ax.set_title('OPT-1.3B  Restore + First Training Step', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=17)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim(0, max(totals) * 1.35)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    counts = ', '.join(
        f'{e[0].split(chr(10))[0]} n={len(df)}'
        for e, df in zip(entries, [st_df, cf_df, pl_df]) if df is not None
    )
    ax.text(0.98, 0.98, counts, transform=ax.transAxes,
            fontsize=11, va='top', ha='right', color='grey')

    plt.tight_layout()
    path = os.path.join(output_dir, 'recovery_restore_first_step.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ================================================================
#  Figure 2 – Full recovery stage breakdown (horizontal stacked bar)
# ================================================================
def plot_recovery_breakdown(pl_df, st_df, cf_df, output_dir):
    """Horizontal stacked bar: breakdown of the full recovery pipeline."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Define stages per method  (col_name, display_label)
    common_stages = [
        ('initialization',            'Init'),
        ('dataset_loading',           'Dataset'),
        ('model_loading',             'Model Load'),
        ('accelerator_preparation',   'Accelerator'),
    ]
    method_specs = [
        ('Standard PyTorch', st_df, C_STANDARD,
         common_stages + [('checkpoint_resume', 'Ckpt Load'), ('first_training_step', '1st Step')]),
        ('CheckFreq',        cf_df, C_CHECKFREQ,
         common_stages + [('checkfreq_restore', 'CF Restore'), ('first_training_step', '1st Step')]),
        ('PipeLayer',        pl_df, C_PIPELAYER,
         common_stages + [('pipelayer_setup', 'PL Setup'), ('first_training_step', '1st Step')]),
    ]

    # Collect a unified ordered set of display labels
    all_labels_ordered = []
    for _, _, _, stages in method_specs:
        for _, lbl in stages:
            if lbl not in all_labels_ordered:
                all_labels_ordered.append(lbl)

    palette = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12',
               '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(all_labels_ordered)}

    bar_data = []   # list of dicts  {label: value}
    bar_labels = [] # method names

    for name, df, _, stages in method_specs:
        if df is None:
            continue
        row = {}
        for col, lbl in stages:
            if col in df.columns:
                row[lbl] = df[col].dropna().mean()
            else:
                row[lbl] = 0
        bar_data.append(row)
        bar_labels.append(name)

    if not bar_data:
        print("No data for breakdown chart")
        return

    y = np.arange(len(bar_labels))
    height = 0.50
    bottoms = np.zeros(len(bar_labels))

    drawn_labels = set()
    for lbl in all_labels_ordered:
        vals = [d.get(lbl, 0) for d in bar_data]
        if max(vals) == 0:
            continue
        label_for_legend = lbl if lbl not in drawn_labels else None
        drawn_labels.add(lbl)
        ax.barh(y, vals, left=bottoms, height=height,
                label=label_for_legend, color=color_map[lbl],
                edgecolor='white', linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 1.2:
                ax.text(bottoms[i] + v / 2, y[i], f'{v:.1f}',
                        ha='center', va='center', fontsize=11,
                        fontweight='bold', color='white')
        bottoms += vals

    for i, total in enumerate(bottoms):
        ax.text(total + 0.3, y[i], f'{total:.1f}s',
                va='center', fontsize=15, fontweight='bold')

    ax.set_xlabel('Time (seconds)', fontsize=18)
    ax.set_yticks(y)
    ax.set_yticklabels(bar_labels, fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_title('OPT-1.3B  Recovery Pipeline Breakdown', fontsize=20)
    ax.legend(loc='upper right', fontsize=11, ncol=3)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'recovery_breakdown.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ================================================================
#  Figure 3 – Pure checkpoint load time
# ================================================================
def plot_checkpoint_load_time(pl_df, st_df, cf_df, output_dir):
    """Bar chart: pure checkpoint / restore loading time."""
    fig, ax = plt.subplots(figsize=(9, 5))

    methods = []
    means = []
    stds_v = []
    colors = []

    for label, df, col, color in [
        ('Standard\n(torch.load)',   st_df, 'checkpoint_resume',  C_STANDARD),
        ('CheckFreq\n(torch.load→GPU)', cf_df, 'checkfreq_restore', C_CHECKFREQ),
        ('PipeLayer\n(mmap read)',   pl_df, 'pipelayer_setup',    C_PIPELAYER),
    ]:
        if df is not None and col in df.columns:
            vals = df[col].dropna()
            methods.append(label)
            means.append(vals.mean())
            stds_v.append(vals.std() if len(vals) > 1 else 0)
            colors.append(color)

    if not methods:
        print("No checkpoint load time data available")
        return

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds_v, width=0.45, color=colors,
                  edgecolor='black', linewidth=1.2, capsize=6,
                  error_kw={'linewidth': 2})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f'{mean:.2f}s', ha='center', va='bottom',
                fontsize=17, fontweight='bold')

    ax.set_title('Pure Checkpoint Load / Restore Time', fontsize=20)
    ax.set_ylabel('Time (s)', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, max(means) * 1.35 if means else 30)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'checkpoint_load_time.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ================================================================
#  Summary printout
# ================================================================
def print_summary(pl_df, st_df, cf_df):
    """Print statistical summary focusing on restore + first step."""
    print("\n" + "=" * 70)
    print("  Recovery Benchmark Summary  —  OPT-1.3B")
    print("=" * 70)

    results = {}  # method -> (restore_mean, step_mean)

    for name, df, method in [
        ('Standard PyTorch', st_df, 'standard'),
        ('CheckFreq',        cf_df, 'checkfreq'),
        ('PipeLayer',        pl_df, 'pipelayer'),
    ]:
        vals = _get_restore_and_first_step(df, method)
        if vals is None:
            continue
        r_mean, r_std, s_mean, s_std = vals
        n = len(df)
        total = r_mean + s_mean
        results[name] = (r_mean, s_mean)

        restore_col = {'standard': 'checkpoint_resume',
                       'pipelayer': 'pipelayer_setup',
                       'checkfreq': 'checkfreq_restore'}[method]
        print(f"\n  {name}  (n={n})")
        print(f"    {restore_col:25s}: {r_mean:7.2f} ± {r_std:.2f} s")
        print(f"    {'first_training_step':25s}: {s_mean:7.2f} ± {s_std:.2f} s")
        print(f"    {'restore + first_step':25s}: {total:7.2f} s")

    # Pairwise speedups
    if len(results) > 1:
        print("\n  ── Speedup (restore + first step) ──")
        names = list(results.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                t_i = sum(results[names[i]])
                t_j = sum(results[names[j]])
                if t_j > 0 and t_i > 0:
                    if t_i > t_j:
                        print(f"    {names[j]} is {t_i / t_j:.2f}× faster than {names[i]}")
                    else:
                        print(f"    {names[i]} is {t_j / t_i:.2f}× faster than {names[j]}")

    print("=" * 70)


# ================================================================
#  Main
# ================================================================
if __name__ == "__main__":
    result_dir = sys.argv[1] if len(sys.argv) > 1 else None

    pl_df, st_df, cf_df = load_data(result_dir)

    if pl_df is None and st_df is None and cf_df is None:
        print("ERROR: No data found. Check CSV paths.")
        sys.exit(1)

    output_dir = (result_dir if result_dir
                  else os.path.join(HOME, "download/pipelayer/examples"))
    os.makedirs(output_dir, exist_ok=True)

    print_summary(pl_df, st_df, cf_df)
    plot_first_loop_comparison(pl_df, st_df, cf_df, output_dir)
    plot_restore_first_step(pl_df, st_df, cf_df, output_dir)
    plot_recovery_breakdown(pl_df, st_df, cf_df, output_dir)
    plot_checkpoint_load_time(pl_df, st_df, cf_df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")
