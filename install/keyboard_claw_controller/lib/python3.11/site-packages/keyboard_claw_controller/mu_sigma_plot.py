"""
plot_grip_results_distribution.py
----------------------------------
Usage:
    python plot_grip_results_distribution.py                      # auto-picks most recent CSV
    python plot_grip_results_distribution.py /path/to/file.csv    # explicit file

Requires CSVs produced by the updated strength_test.py that includes
force_mu and force_sigma columns.

Produces four figures:
    1. IR success rate per (mu, sigma) distribution  (bar chart)
    2. IR success rate vs mu, one line per sigma      (line + markers)
    3. Actual drawn force distribution vs success     (scatter + gaussian overlay)
    4. Trial-by-trial success, coloured by (mu, sigma) group
"""

import sys
import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list:
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            for col in ('success_rfid', 'success_ir', 'success_combined'):
                if col in row:
                    row[col] = row[col].lower() in ('true', '1', 'yes')
            for col in ('force', 'force_mu', 'force_sigma',
                        'start_position_x', 'start_position_y',
                        'trial_number', 'duration'):
                if col in row and row[col] not in (None, ''):
                    try:
                        row[col] = float(row[col])
                    except (ValueError, TypeError):
                        row[col] = None
            rows.append(row)
    return rows


def find_latest_csv() -> str:
    search_dirs = ['.', 'claw_data', '/home/clawMachine/claw_data']
    candidates = []
    for d in search_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith('grip_strength_test') and f.endswith('.csv'):
                    candidates.append(os.path.join(d, f))
    return sorted(candidates)[-1] if candidates else None


def dist_label(mu, sigma):
    return f"mu={mu:.4g}, sigma={sigma:.4g}"


def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ── plot 1 : IR success rate per (mu, sigma) distribution ────────────────────

def plot_by_distribution(rows: list, ax: plt.Axes):
    """Bar chart: one bar per unique (mu, sigma) combination."""
    buckets = defaultdict(list)
    for r in rows:
        key = (r.get('force_mu'), r.get('force_sigma'))
        buckets[key].append(r['success_ir'])

    labels    = sorted(buckets.keys(), key=lambda k: (k[0] or 0, k[1] or 0))
    xlabels   = [dist_label(*k) for k in labels]
    counts    = [len(buckets[k]) for k in labels]
    successes = [int(sum(buckets[k])) for k in labels]
    rates     = [s / n * 100 for s, n in zip(successes, counts)]

    colors = plt.cm.viridis([i / max(len(labels) - 1, 1) for i in range(len(labels))])
    bars = ax.bar(xlabels, rates, color=colors, edgecolor='white', linewidth=1.2, zorder=3)

    for bar, rate, s, n in zip(bars, rates, successes, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{s}/{n}\n{rate:.0f}%",
                ha='center', va='bottom', fontsize=9, color='#333333')

    ax.set_title("IR Success Rate by Distribution (mu, sigma)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Distribution", fontsize=10)
    ax.set_ylabel("IR Success Rate (%)", fontsize=10)
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right', fontsize=9)


# ── plot 2 : IR success rate vs mu, one line per sigma ───────────────────────

def plot_mu_vs_success_by_sigma(rows: list, ax: plt.Axes):
    """Line plot: x=mu, y=success rate, separate line per sigma value."""
    sigmas = sorted({r.get('force_sigma') for r in rows if r.get('force_sigma') is not None})
    colors = plt.cm.tab10([i / max(len(sigmas) - 1, 1) for i in range(len(sigmas))])

    for sigma, color in zip(sigmas, colors):
        sigma_rows = [r for r in rows if r.get('force_sigma') == sigma]
        mus = sorted({r['force_mu'] for r in sigma_rows if r.get('force_mu') is not None})

        rates, successes, counts = [], [], []
        for mu in mus:
            subset = [r for r in sigma_rows if r['force_mu'] == mu]
            n = len(subset)
            s = int(sum(r['success_ir'] for r in subset))
            counts.append(n)
            successes.append(s)
            rates.append(s / n * 100 if n else 0)

        se = [((r / 100 * (1 - r / 100)) / n) ** 0.5 * 100 if n else 0
              for r, n in zip(rates, counts)]
        lo = [max(0,   r - e) for r, e in zip(rates, se)]
        hi = [min(100, r + e) for r, e in zip(rates, se)]

        ax.plot(mus, rates, color=color, linewidth=2,
                marker='o', markersize=8,
                markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=2, zorder=3, label=f"sigma={sigma:.4g}")
        ax.fill_between(mus, lo, hi, color=color, alpha=0.12, zorder=2)

        for mu, r, s, n in zip(mus, rates, successes, counts):
            ax.annotate(f"{s}/{n}\n{r:.0f}%",
                        xy=(mu, r), xytext=(0, 14),
                        textcoords='offset points',
                        ha='center', fontsize=9, color='#333333')

    ax.set_title("IR Success Rate vs mu  (line per sigma)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Force mu", fontsize=10)
    ax.set_ylabel("IR Success Rate (%)", fontsize=10)
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, title="Sigma", title_fontsize=9)


# ── plot 3 : actual drawn force vs success + gaussian overlay ─────────────────

def plot_drawn_force_vs_success(rows: list, ax: plt.Axes):
    """
    Scatter of actual drawn force (x) vs success (y=1/0).
    Behind the scatter, draw the theoretical Gaussian PDF for each (mu, sigma)
    group, scaled to fit the y-axis range so it's visible as a shape reference.
    """
    groups = sorted({(r.get('force_mu'), r.get('force_sigma'))
                     for r in rows
                     if r.get('force_mu') is not None and r.get('force_sigma') is not None},
                    key=lambda k: (k[0], k[1]))

    colors = plt.cm.tab10([i / max(len(groups) - 1, 1) for i in range(len(groups))])

    all_forces = [r['force'] for r in rows if r.get('force') is not None]
    x_min = min(all_forces) - 1
    x_max = max(all_forces) + 1
    x_curve = np.linspace(x_min, x_max, 400)

    for (mu, sigma), color in zip(groups, colors):
        subset = [r for r in rows
                  if r.get('force_mu') == mu and r.get('force_sigma') == sigma]

        forces  = [r['force'] for r in subset]
        success = [1 if r['success_ir'] else 0 for r in subset]

        rng = np.random.default_rng(seed=int(mu * 10 + (sigma or 0) * 100))
        jitter = rng.uniform(-0.03, 0.03, size=len(success))

        ax.scatter(forces, [y + j for y, j in zip(success, jitter)],
                   color=color, s=60, edgecolors='white', linewidths=0.6,
                   zorder=4, label=dist_label(mu, sigma))

        # Gaussian curve scaled so peak sits at y=1.15 (above the pass line)
        if sigma and sigma > 0:
            pdf = gaussian(x_curve, mu, sigma)
            pdf_scaled = pdf / pdf.max() * 0.4 + 0.8   # scale peak to y≈1.2
            ax.plot(x_curve, pdf_scaled, color=color,
                    linewidth=1.5, linestyle='--', alpha=0.7, zorder=2)

    ax.axhline(0.5, color='#aaaaaa', linewidth=0.8, linestyle=':', zorder=1)
    ax.set_title("Drawn Force vs IR Success  (dashed = Gaussian shape)", fontsize=12,
                 fontweight='bold', pad=10)
    ax.set_xlabel("Actual drawn force value", fontsize=10)
    ax.set_ylabel("IR success  (1 = pass, 0 = fail)", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail (0)", "Pass (1)"])
    ax.set_ylim(-0.25, 1.45)
    ax.set_xlim(x_min, x_max)
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, loc='lower right')


# ── plot 4 : trial-by-trial success coloured by (mu, sigma) group ─────────────

def plot_trial_vs_success(rows: list, ax: plt.Axes):
    """
    Scatter of trial number vs success, colour/shape = (mu, sigma) group.
    Rolling average overlaid.
    """
    groups = sorted({(r.get('force_mu'), r.get('force_sigma'))
                     for r in rows
                     if r.get('force_mu') is not None and r.get('force_sigma') is not None},
                    key=lambda k: (k[0], k[1]))

    colors  = plt.cm.tab10([i / max(len(groups) - 1, 1) for i in range(len(groups))])
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v']

    sorted_rows = sorted(rows, key=lambda r: r['trial_number'])

    for gi, ((mu, sigma), color) in enumerate(zip(groups, colors)):
        subset = [r for r in sorted_rows
                  if r.get('force_mu') == mu and r.get('force_sigma') == sigma]
        xs = [r['trial_number'] for r in subset]
        ys = [1 if r['success_ir'] else 0 for r in subset]

        rng = np.random.default_rng(seed=gi)
        jitter = rng.uniform(-0.03, 0.03, size=len(ys))

        ax.scatter(xs, [y + j for y, j in zip(ys, jitter)],
                   marker=markers[gi % len(markers)],
                   color=color, s=70,
                   edgecolors='white', linewidths=0.6,
                   zorder=4, label=dist_label(mu, sigma))

    # Rolling average across all trials
    all_xs = [r['trial_number'] for r in sorted_rows]
    all_ys = [1 if r['success_ir'] else 0 for r in sorted_rows]
    window = min(5, len(all_ys))
    if window > 1:
        kernel  = np.ones(window) / window
        rolling = np.convolve(all_ys, kernel, mode='valid')
        roll_xs = all_xs[window - 1:]
        ax.plot(roll_xs, rolling, color='#333333', linewidth=1.8,
                linestyle='--', zorder=3, label=f"Rolling avg (n={window})")

    # Vertical line each time (mu, sigma) group changes
    prev_group = None
    for r in sorted_rows:
        g = (r.get('force_mu'), r.get('force_sigma'))
        if g != prev_group:
            if prev_group is not None:
                ax.axvline(r['trial_number'] - 0.5,
                           color='#aaaaaa', linewidth=0.8, linestyle=':', zorder=1)
            prev_group = g

    ax.set_title("IR success per trial  (colour = distribution group)", fontsize=12,
                 fontweight='bold', pad=10)
    ax.set_xlabel("Trial number", fontsize=10)
    ax.set_ylabel("IR success  (1 = pass, 0 = fail)", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail (0)", "Pass (1)"])
    ax.set_ylim(-0.25, 1.35)
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, loc='upper right', ncol=2)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        csv_path = find_latest_csv()
        if csv_path is None:
            print("Usage: python plot_grip_results_distribution.py <path_to_csv>")
            print("       (no grip_strength_test*.csv found automatically)")
            sys.exit(1)
        print(f"[INFO] No file specified — using most recent: {csv_path}")

    if not os.path.isfile(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    rows = load_csv(csv_path)
    print(f"[INFO] Loaded {len(rows)} trials from {csv_path}")

    # Warn if mu/sigma columns are missing (old CSV format)
    if rows and 'force_mu' not in rows[0]:
        print("[WARN] force_mu / force_sigma columns not found — "
              "make sure the CSV was produced by the updated strength_test.py")
        sys.exit(1)

    missing = [c for c in ('success_ir', 'force', 'force_mu', 'force_sigma',
                           'start_position_x', 'start_position_y')
               if rows and c not in rows[0]]
    if missing:
        print(f"[ERROR] CSV is missing columns: {missing}")
        sys.exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("Grip Strength Test — Results by Distribution (mu / sigma)",
                 fontsize=15, fontweight='bold', y=1.01)

    plot_by_distribution(rows, axes[0, 0])
    plot_mu_vs_success_by_sigma(rows, axes[0, 1])
    plot_drawn_force_vs_success(rows, axes[1, 0])
    plot_trial_vs_success(rows, axes[1, 1])

    plt.tight_layout()

    out_path = csv_path.replace('.csv', '_distribution_plots.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved figure → {out_path}")
    plt.show()


if __name__ == '__main__':
    main()