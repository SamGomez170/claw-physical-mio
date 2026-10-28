"""
Success rate per grip distribution type, from the trials you actually ran,
counting ONLY the trials inside the participants' aim area.

For each entry in grip_distribution_types below, it takes the grip-test trials
that were run with exactly that mu and sigma (claw_mu / claw_sigma columns),
keeps those inside the aim area (aim_areas.json from goal_heatmap.py),
and plots  successes / trials  as a bar. No fitting, no model.

>>> Edit the settings below, then run:   python claw_bars_aim_area.py
"""
import csv
import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# SETTINGS - edit these
# =====================================================================
grip_distribution_types = {
    "wide_high": {
        "force": {"mu": 199, "sigma": 2},
        "rate": 255
    },
    "narrow_high": {
        "force": {"mu": 199, "sigma": 0.5},
        "rate": 255
    },
    "wide_low": {
        "force": {"mu": 191, "sigma": 2},
        "rate": 255
    },
    "narrow_low": {
        "force": {"mu": 191, "sigma": 0.5},
        "rate": 255
    }
}

DATA_FOLDER = "/home/clawMachine/claw_data/offset_data"   # grip-test .csv / .json files
AREAS_FILE = os.path.join(DATA_FOLDER, "aim_areas.json")  # made by goal_heatmap.py
OUTPUT_PNG = os.path.join(DATA_FOLDER, "success_by_claw_aim_area.png")
# =====================================================================

GOALS = {"Goal 1": (750.0, 450.0), "Goal 2": (625.0, 150.0), "Goal 3": (445.0, 385.0)}


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _records(path):
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def in_area(r, areas):
    bx, by = _num(r.get("ball_x")), _num(r.get("ball_y"))
    if bx is None or by is None:
        return False
    mx, my = _num(r.get("grab_measured_x")), _num(r.get("grab_measured_y"))
    if mx is not None and my is not None:
        dx, dy = mx - bx, my - by
    else:                                         # no measured position saved
        dx, dy = _num(r.get("offset_dx")) or 0.0, _num(r.get("offset_dy")) or 0.0
    name = min(GOALS, key=lambda g: math.hypot(bx - GOALS[g][0], by - GOALS[g][1]))
    a = areas.get(name)
    if a is None:
        return False
    d = np.array([dx, dy]) - np.array(a["mean"])
    return float(d @ np.linalg.inv(np.array(a["cov"])) @ d) <= a["r2"]


def load_trials(folder):
    files = []
    for ext in ("csv", "json"):
        files += glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True)
    trials = {}
    for path in sorted(set(files)):
        try:
            recs = _records(path)
        except Exception:
            continue
        for r in recs:
            if isinstance(r, dict) and "success_ir" in r and _num(r.get("claw_mu")) is not None:
                trials[r.get("timestamp") or (path, r.get("trial_number"))] = r
    return list(trials.values())


def main():
    if not os.path.isfile(AREAS_FILE):
        raise SystemExit(f"No aim areas file: {AREAS_FILE}\nRun offsets.py on the participant folder first.")
    with open(AREAS_FILE) as f:
        areas = json.load(f)
    trials = load_trials(DATA_FOLDER)

    names, labels, k_hit, n_tot, n_all = [], [], [], [], []
    print("claw            mu    sigma   inside area      all trials")
    for name, c in grip_distribution_types.items():
        mu, sd = c["force"]["mu"], c["force"]["sigma"]
        same = [r for r in trials
                if abs(_num(r["claw_mu"]) - mu) < 1e-6 and abs(_num(r["claw_sigma"]) - sd) < 1e-6]
        inside = [r for r in same if in_area(r, areas)]
        k = sum(str(r["success_ir"]).strip().lower() == "true" for r in inside)
        names.append(name)
        labels.append(f"{name.replace('_', ' ').title()}\n(μ={mu:g}, σ={sd:g})")
        k_hit.append(k); n_tot.append(len(inside)); n_all.append(len(same))
        rate = f"{k}/{len(inside)} = {k/len(inside):.1%}" if inside else "no trials"
        print(f"{name:14s} {mu:5g} {sd:6g}   {rate:16s} {len(same)}")

    rates = [100 * k / n if n else 0 for k, n in zip(k_hit, n_tot)]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    x = np.arange(len(names))
    ax.bar(x, rates, width=0.5, color="#3b9ff5", edgecolor="black", linewidth=1.5, zorder=3)
    for i in range(len(names)):
        txt = f"{rates[i]:.1f}%\n({k_hit[i]}/{n_tot[i]})" if n_tot[i] else "no trials"
        ax.text(x[i], rates[i] + 2, txt, ha="center", va="bottom", fontsize=12, weight="bold")
    ax.set_xticks(x, labels, fontsize=12, weight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_ylabel("Success Rate", fontsize=14, weight="bold")
    ax.grid(axis="y", color="#e4e3de", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cov = next(iter(areas.values()))["coverage"]
    ax.set_title(f"Success Rate by Grip Distribution Type (trials inside aim area, {cov:.0%})",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, facecolor="white")
    print(f"\nsaved {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()