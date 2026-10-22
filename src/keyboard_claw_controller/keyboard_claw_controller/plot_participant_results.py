"""
plot_force_vs_success.py
------------------------
Produces two side-by-side figures based on IR detection (not RFID/success field):

    1. Grip Force vs IR Detection Rate  (bar chart with error bars)
    2. IR Detection per Trial           (scatter + rolling average, shape = force)

Auto-discovers the most recent combined_all_trials_*.json under
/home/clawMachine/claw_data/<participant>/session_*/

Usage:
    python plot_force_vs_success.py                     # auto-picks latest
    python plot_force_vs_success.py /path/to/file.json  # explicit file
"""

import sys
import os
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def ir_success(trial: dict) -> bool:
    """True if IR detected something (any string other than 'No IR detection')."""
    return trial.get("ir_detection", "").strip().lower() != "no ir detection"


def find_latest_json() -> str:
    base    = "/home/clawMachine/claw_data"
    pattern = os.path.join(base, "*", "session_*", "combined_all_trials_*.json")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def load_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)


# ── plot 1 : grip force vs IR detection rate (bar chart) ─────────────────────

def plot_force_vs_success(trials: list, ax: plt.Axes):
    buckets = defaultdict(list)
    for t in trials:
        buckets[t["grip_force"]].append(ir_success(t))

    forces    = sorted(buckets.keys())
    counts    = [len(buckets[f]) for f in forces]
    successes = [sum(buckets[f]) for f in forces]
    rates     = [s / n * 100 for s, n in zip(successes, counts)]

    se = [((r / 100 * (1 - r / 100)) / n) ** 0.5 * 100 if n else 0
          for r, n in zip(rates, counts)]

    colors = [plt.cm.RdYlGn(r / 100) for r in rates]

    x    = range(len(forces))
    bars = ax.bar(x, rates, color=colors, edgecolor="white",
                  linewidth=1.2, zorder=3, yerr=se,
                  capsize=5, error_kw=dict(ecolor="#555", elinewidth=1.2))

    max_se = max(se) if se else 0
    for bar, rate, s, n in zip(bars, rates, successes, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_se + 2,
            f"{s}/{n}\n{rate:.0f}%",
            ha="center", va="bottom", fontsize=9, color="#333333",
        )

    ax.set_title("Grip Force vs IR Detection Rate",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Grip Force (N)", fontsize=10)
    ax.set_ylabel("IR Detection Rate (%)", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [str(int(f)) if f == int(f) else str(f) for f in forces]
    )
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# ── plot 2 : trial number vs IR success, shape = force level ─────────────────

def plot_trial_vs_success(trials: list, ax: plt.Axes):
    sorted_trials = sorted(trials, key=lambda t: t["trial_number"])

    forces  = sorted({t["grip_force"] for t in sorted_trials})
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    colors  = plt.cm.tab10([i / max(len(forces) - 1, 1) for i in range(len(forces))])

    for fi, (force, color, marker) in enumerate(zip(forces, colors, markers)):
        subset = [t for t in sorted_trials if t["grip_force"] == force]
        xs = [t["trial_number"] for t in subset]
        ys = [1 if ir_success(t) else 0 for t in subset]

        rng    = np.random.default_rng(seed=fi)
        jitter = rng.uniform(-0.03, 0.03, size=len(ys))

        ax.scatter(xs, [y + j for y, j in zip(ys, jitter)],
                   marker=marker, color=color, s=70,
                   edgecolors="white", linewidths=0.6,
                   zorder=4, label=f"Force {force:.0f}")

    # Rolling average across all trials
    all_xs = [t["trial_number"] for t in sorted_trials]
    all_ys = [1 if ir_success(t) else 0 for t in sorted_trials]
    window = min(5, len(all_ys))
    if window > 1:
        kernel  = np.ones(window) / window
        rolling = np.convolve(all_ys, kernel, mode="valid")
        roll_xs = all_xs[window - 1:]
        ax.plot(roll_xs, rolling, color="#333333", linewidth=1.8,
                linestyle="--", zorder=3, label=f"Rolling avg (n={window})")

    # Faint vertical lines at force-level transitions
    prev_force = None
    for t in sorted_trials:
        if t["grip_force"] != prev_force:
            if prev_force is not None:
                ax.axvline(t["trial_number"] - 0.5,
                           color="#aaaaaa", linewidth=0.8,
                           linestyle=":", zorder=1)
            prev_force = t["grip_force"]

    ax.set_title("IR Detection per Trial  (shape = force level)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Trial number", fontsize=10)
    ax.set_ylabel("IR detection  (1 = detected, 0 = not)", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not detected (0)", "Detected (1)"])
    ax.set_ylim(-0.25, 1.35)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", ncol=2)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) >= 2:
        json_path = sys.argv[1]
    else:
        json_path = find_latest_json()
        if json_path is None:
            print("[ERROR] No combined_all_trials_*.json found under "
                  "/home/clawMachine/claw_data/")
            print("Usage: python plot_force_vs_success.py <path_to_json>")
            sys.exit(1)
        print(f"[INFO] No file specified — using most recent: {json_path}")

    if not os.path.isfile(json_path):
        print(f"[ERROR] File not found: {json_path}")
        sys.exit(1)

    trials = load_json(json_path)
    print(f"[INFO] Loaded {len(trials)} trials from {json_path}")

    parts = json_path.replace("\\", "/").split("/")
    try:
        subtitle = f"{parts[-3]} / {parts[-2]}"
    except IndexError:
        subtitle = os.path.basename(json_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"IR Detection Results  —  {subtitle}",
                 fontsize=14, fontweight="bold", y=1.01)

    plot_force_vs_success(trials, ax1)
    plot_trial_vs_success(trials, ax2)

    plt.tight_layout()
    out_path = json_path.replace(".json", "_force_vs_success.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved figure → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()