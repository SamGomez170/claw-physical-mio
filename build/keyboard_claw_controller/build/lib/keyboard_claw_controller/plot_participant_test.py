"""
plot_force_vs_success.py
------------------------
Produces two side-by-side figures based on IR detection (not RFID/success field).
Groups trials by (force_mu, force_sigma) instead of raw grip_force.

    1. (mu, sigma) distribution vs IR Detection Rate  (bar chart with error bars)
    2. IR Detection per Trial  (scatter + rolling average, shape = distribution group)

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


def dist_key(trial: dict):
    """Return (force_mu, force_sigma) tuple for grouping."""
    return (trial.get("force_mu"), trial.get("force_sigma"))


def dist_label(mu, sigma) -> str:
    mu_str    = str(int(mu))    if mu    == int(mu)    else str(mu)
    sigma_str = str(int(sigma)) if sigma == int(sigma) else str(sigma)
    return f"mu={mu_str}\nsigma={sigma_str}"


def dist_label_inline(mu, sigma) -> str:
    """Single-line version for legend entries."""
    mu_str    = str(int(mu))    if mu    == int(mu)    else str(mu)
    sigma_str = str(int(sigma)) if sigma == int(sigma) else str(sigma)
    return f"mu={mu_str}, sigma={sigma_str}"


def find_latest_json() -> str:
    base      = "/home/clawMachine/claw_data"
    pattern   = os.path.join(base, "*", "session_*", "combined_all_trials_*.json")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def load_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)


# ── plot 1 : (mu, sigma) vs IR detection rate (bar chart) ────────────────────

def plot_dist_vs_success(trials: list, ax: plt.Axes):
    buckets = defaultdict(list)
    for t in trials:
        buckets[dist_key(t)].append(ir_success(t))

    groups    = sorted(buckets.keys(), key=lambda k: (k[0] or 0, k[1] or 0))
    counts    = [len(buckets[g]) for g in groups]
    successes = [sum(buckets[g]) for g in groups]
    rates     = [s / n * 100 for s, n in zip(successes, counts)]
    se        = [((r / 100 * (1 - r / 100)) / n) ** 0.5 * 100 if n else 0
                 for r, n in zip(rates, counts)]

    colors  = [plt.cm.RdYlGn(r / 100) for r in rates]
    xlabels = [dist_label(*g) for g in groups]
    x       = range(len(groups))

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

    ax.set_title("Distribution (mu, sigma) vs IR Detection Rate",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Force Distribution", fontsize=10)
    ax.set_ylabel("IR Detection Rate (%)", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# ── plot 2 : trial number vs IR success, shape = distribution group ───────────

def plot_trial_vs_success(trials: list, ax: plt.Axes):
    sorted_trials = sorted(trials, key=lambda t: t["trial_number"])

    groups  = sorted({dist_key(t) for t in sorted_trials},
                     key=lambda k: (k[0] or 0, k[1] or 0))
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    colors  = plt.cm.tab10([i / max(len(groups) - 1, 1) for i in range(len(groups))])

    for gi, (group, color) in enumerate(zip(groups, colors)):
        subset = [t for t in sorted_trials if dist_key(t) == group]
        xs = [t["trial_number"] for t in subset]
        ys = [1 if ir_success(t) else 0 for t in subset]

        rng    = np.random.default_rng(seed=gi)
        jitter = rng.uniform(-0.03, 0.03, size=len(ys))

        ax.scatter(xs, [y + j for y, j in zip(ys, jitter)],
                   marker=markers[gi % len(markers)],
                   color=color, s=70,
                   edgecolors="white", linewidths=0.6,
                   zorder=4, label=dist_label_inline(*group))

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

    # Faint vertical lines at distribution group transitions
    prev_group = None
    for t in sorted_trials:
        g = dist_key(t)
        if g != prev_group:
            if prev_group is not None:
                ax.axvline(t["trial_number"] - 0.5,
                           color="#aaaaaa", linewidth=0.8,
                           linestyle=":", zorder=1)
            prev_group = g

    ax.set_title("IR Detection per Trial  (shape = distribution group)",
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

    # Warn if mu/sigma keys are missing
    sample = trials[0] if trials else {}
    if "force_mu" not in sample or "force_sigma" not in sample:
        print("[WARN] force_mu / force_sigma keys not found in JSON. "
              "Make sure the data was produced by the updated strength_test.py.")
        sys.exit(1)

    parts = json_path.replace("\\", "/").split("/")
    try:
        subtitle = f"{parts[-3]} / {parts[-2]}"
    except IndexError:
        subtitle = os.path.basename(json_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"IR Detection Results  —  {subtitle}",
                 fontsize=14, fontweight="bold", y=1.01)

    plot_dist_vs_success(trials, ax1)
    plot_trial_vs_success(trials, ax2)

    plt.tight_layout()
    out_path = json_path.replace(".json", "_dist_vs_success.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved figure → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
    