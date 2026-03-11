"""
plot_grip_results.py
--------------------
Usage:
    python plot_grip_results.py                          # auto-picks most recent CSV
    python plot_grip_results.py /path/to/file.csv        # explicit file

Produces three figures:
    1. IR success rate vs start position  (bar chart)
    2. IR success rate vs force           (line + markers)
    3. IR success rate vs force, grouped bars coloured by position
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
    """Load CSV into a list of dicts with normalised types."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            for col in ('success_rfid', 'success_ir', 'success_combined'):
                if col in row:
                    row[col] = row[col].lower() in ('true', '1', 'yes')
            for col in ('force', 'start_position_x', 'start_position_y',
                        'trial_number', 'duration'):
                if col in row:
                    try:
                        row[col] = float(row[col])
                    except (ValueError, TypeError):
                        row[col] = None
            rows.append(row)
    return rows


def find_latest_csv() -> str:
    """Search common locations for the most recent grip_strength_test CSV."""
    search_dirs = ['.', 'claw_data', '/home/clawMachine/claw_data']
    candidates = []
    for d in search_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith('grip_strength_test') and f.endswith('.csv'):
                    candidates.append(os.path.join(d, f))
    return sorted(candidates)[-1] if candidates else None


# ── plot 1 : IR success rate vs start position ────────────────────────────────

def plot_by_position(rows: list, ax: plt.Axes):
    buckets = defaultdict(list)
    for r in rows:
        key = (int(r['start_position_x']), int(r['start_position_y']))
        buckets[key].append(r['success_ir'])

    labels    = sorted(buckets.keys(), key=lambda p: str(p))
    xlabels   = [f"({k[0]}, {k[1]})" for k in labels]
    counts    = [len(buckets[k]) for k in labels]
    successes = [int(sum(buckets[k])) for k in labels]
    rates     = [s / n * 100 for s, n in zip(successes, counts)]

    colors = plt.cm.viridis([i / max(len(labels) - 1, 1)
                              for i in range(len(labels))])
    bars = ax.bar(xlabels, rates, color=colors,
                  edgecolor='white', linewidth=1.2, zorder=3)

    for bar, rate, s, n in zip(bars, rates, successes, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{s}/{n}\n{rate:.0f}%",
                ha='center', va='bottom', fontsize=9, color='#333333')

    ax.set_title("IR Success Rate by Start Position", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Start Position (x, y)  [mm]", fontsize=10)
    ax.set_ylabel("IR Success Rate (%)", fontsize=10)
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


# ── plot 2 : IR success rate vs force (line) ──────────────────────────────────

def plot_by_force(rows: list, ax: plt.Axes):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r['force']].append(r['success_ir'])

    forces    = sorted(buckets.keys())
    counts    = [len(buckets[f]) for f in forces]
    successes = [int(sum(buckets[f])) for f in forces]
    rates     = [s / n * 100 for s, n in zip(successes, counts)]

    se = [((r / 100 * (1 - r / 100)) / n) ** 0.5 * 100
          for r, n in zip(rates, counts)]
    lo = [max(0,   r - e) for r, e in zip(rates, se)]
    hi = [min(100, r + e) for r, e in zip(rates, se)]

    ax.plot(forces, rates, color='#2a7fbf', linewidth=2,
            marker='o', markersize=8,
            markerfacecolor='white', markeredgecolor='#2a7fbf',
            markeredgewidth=2, zorder=3, label='IR success rate')
    ax.fill_between(forces, lo, hi,
                    color='#2a7fbf', alpha=0.15, zorder=2, label='±1 SE')

    for f, r, s, n in zip(forces, rates, successes, counts):
        ax.annotate(f"{s}/{n}\n{r:.0f}%",
                    xy=(f, r), xytext=(0, 14),
                    textcoords='offset points',
                    ha='center', fontsize=9, color='#333333')

    ax.set_title("IR Success Rate by Grip Force", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Grip Force", fontsize=10)
    ax.set_ylabel("IR Success Rate (%)", fontsize=10)
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9)


# ── plot 3 : force vs success, grouped bars coloured by position ──────────────

def plot_force_by_position(rows: list, ax: plt.Axes):
    # collect unique sorted forces and positions
    forces    = sorted({r['force'] for r in rows})
    positions = sorted({(int(r['start_position_x']), int(r['start_position_y']))
                        for r in rows}, key=lambda p: str(p))

    n_forces = len(forces)
    n_pos    = len(positions)

    # colour each position distinctly
    pos_colors = plt.cm.tab10([i / max(n_pos - 1, 1) for i in range(n_pos)])

    bar_width  = 0.8 / n_pos          # bars share a slot of width 0.8
    x_base     = np.arange(n_forces)  # one tick per force value

    for pi, (pos, color) in enumerate(zip(positions, pos_colors)):
        offsets   = x_base + (pi - (n_pos - 1) / 2) * bar_width
        rates, successes, counts = [], [], []

        for force in forces:
            subset = [r for r in rows
                      if r['force'] == force
                      and int(r['start_position_x']) == pos[0]
                      and int(r['start_position_y']) == pos[1]]
            n = len(subset)
            s = int(sum(r['success_ir'] for r in subset))
            counts.append(n)
            successes.append(s)
            rates.append(s / n * 100 if n else 0)

        bars = ax.bar(offsets, rates, width=bar_width * 0.9,
                      color=color, edgecolor='white', linewidth=0.8,
                      zorder=3, label=f"({pos[0]}, {pos[1]})")

        for bar, rate, s, n in zip(bars, rates, successes, counts):
            if n == 0:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{s}/{n}",
                    ha='center', va='bottom', fontsize=7.5, color='#333333')

    ax.set_title("IR Success Rate: Force × Position", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Grip Force", fontsize=10)
    ax.set_ylabel("IR Success Rate (%)", fontsize=10)
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(int(f)) if f == int(f) else str(f) for f in forces])
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(title="Start Position (x, y)", fontsize=8, title_fontsize=8,
              loc='lower right')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        csv_path = find_latest_csv()
        if csv_path is None:
            print("Usage: python plot_grip_results.py <path_to_csv>")
            print("       (no grip_strength_test*.csv found automatically)")
            sys.exit(1)
        print(f"[INFO] No file specified — using most recent: {csv_path}")

    if not os.path.isfile(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    rows = load_csv(csv_path)
    print(f"[INFO] Loaded {len(rows)} trials from {csv_path}")

    missing = [c for c in ('success_ir', 'force', 'start_position_x', 'start_position_y')
               if rows and c not in rows[0]]
    if missing:
        print(f"[ERROR] CSV is missing columns: {missing}")
        sys.exit(1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Grip Strength Test — IR Sensor Results",
                 fontsize=15, fontweight='bold', y=1.01)

    plot_by_position(rows, axes[0])
    plot_by_force(rows, axes[1])
    plot_force_by_position(rows, axes[2])

    plt.tight_layout()

    out_path = csv_path.replace('.csv', '_plots.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved figure → {out_path}")
    plt.show()


if __name__ == '__main__':
    main()