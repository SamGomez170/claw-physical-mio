"""
Where did participants put the claw down, compared with the 3 ball positions?

Reads every combined summary (combined_all_training_*.json and
combined_all_trials_*.json, including the *_rebuilt ones) found in the given
folder and its sub-folders, assigns each grab (grab_x, grab_y) to the closest
goal, and makes:

  goal_heatmap_all.png / _training.png / _experiment.png
      top:    arena heat map of all grab positions, the 3 goals, hits and misses
      bottom: one zoom per goal showing the offset of each grab from the goal
  goal_errors.csv    one row per trial: nearest goal, dx, dy, distance, hit
  goal_summary.csv   per phase x goal: n, hit rate, mean offset, mean distance

Usage:
    python goal_heatmap.py /home/clawMachine/claw_data
    python goal_heatmap.py /home/clawMachine/claw_data --out my_plots
    python goal_heatmap.py /home/clawMachine/claw_data/offset_data --tests
        --tests: plot the automatic grip tests (grip_offset_test_*.csv/.json)
        instead; uses the measured grab position, or the target position when
        no measured position was saved.
        Only test trials run with a distribution listed in grip_distribution_types
        (same mu AND sigma) are used, all together in one plot.
        Set ONLY_LISTED_DISTRIBUTIONS = False to use every test trial.

Both modes also make, for the distributions listed in grip_distribution_types
(ALL trials with that mu and sigma, no aim-area filter):
    success_by_distribution[_tests].png   bar plot of the success rate per distribution
    success_by_distribution_phases.png    same, training and experiment side by side
                                          (participants only)
    force_vs_success[_tests].png          every trial at its drawn force, pass (top) /
                                          fail (bottom), coloured by distribution, with
                                          each distribution's Gaussian drawn dashed

Only needs numpy and matplotlib.
"""
import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D

# ---------------- settings ----------------
GOALS = {                      # ball positions (same units as grab_x / grab_y)
    "Goal 1": (750.0, 450.0),
    "Goal 2": (625.0, 150.0),
    "Goal 3": (445.0, 385.0),
}
ARENA_X = (250, 800)           # plotted area (widened automatically if a grab falls outside)
ARENA_Y = (50, 520)
MAX_DIST = 150.0               # grabs farther than this from every goal = "off target"
KDE_BANDWIDTH = 9.0            # smoothing of the arena heat map (arena units): smaller = sharper
ZOOM_BANDWIDTH = 5.0           # same for the per-goal zoom panels
FADE = 1.8                     # >1 fades weak areas to white so the dense spots stand out (1 = linear)
SMALL_POINTS = True            # thin markers so the dark-blue spots are not hidden
ZOOM = 80                      # half-width of each per-goal zoom panel
AREA_COVERAGE = 0.80           # aim area (ellipse) holds this share of the grabs at each goal
AREA_MAX_DIST = 80.0           # grabs farther than this from the goal are not used for the ellipse

# --tests: only use trials run with these distributions (same mu AND sigma)
grip_distribution_types = {
    "wide_high": {
        "force": {"mu": 197, "sigma": 2},
        "rate": 255
    },
    "narrow_high": {
        "force": {"mu": 197, "sigma": 0.5},
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
ONLY_LISTED_DISTRIBUTIONS = True
# --tests: draw the participants' aim area on the test plots if this file exists
# (None = look for aim_areas.json in the data folder)
AIM_AREAS_FILE = None

# colours: sequential blue ramp for density, orange for misses, ink for the rest
HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"])
MISS = "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


# ---------------- loading ----------------
def find_summaries(root):
    pats = ["combined_all_training_*.json", "combined_all_trials_*.json"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(root, p))
        files += glob.glob(os.path.join(root, "**", p), recursive=True)
    return sorted(set(f for f in files if not f.endswith(".tmp")))


def load_rows(root):
    """Return one row per (participant, phase, trial). If the same trial is in
    several summary files (e.g. original + _rebuilt), the one with a grab
    position wins, so nothing is counted twice."""
    rows = {}
    for path in find_summaries(root):
        phase = "training" if "combined_all_training_" in os.path.basename(path) else "experiment"
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        for r in data:
            key = (r.get("participant_id"), phase, r.get("trial_number"))
            gx, gy = r.get("grab_x"), r.get("grab_y")
            if gx is None or gy is None:
                rows.setdefault(key, None)
                continue
            rows[key] = {
                "participant_id": key[0], "phase": phase,
                "trial_number": key[2], "grip_type": r.get("grip_type"),
                "grab_x": float(gx), "grab_y": float(gy),
                "hit": bool(r.get("ir_success")),
            }
    return [r for r in rows.values() if r is not None]


def _true(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def match_distribution(r):
    """Name of the listed distribution this test trial was run with, or None."""
    mu, sd = _num(r.get("claw_mu")), _num(r.get("claw_sigma"))
    if mu is None or sd is None:
        return None
    for name, c in grip_distribution_types.items():
        if abs(mu - c["force"]["mu"]) < 1e-6 and abs(sd - c["force"]["sigma"]) < 1e-6:
            return name
    return None


def load_participant_outcomes(root):
    """Every participant trial (also those without a grab position):
    one (distribution name, hit, drawn force, phase) per (participant, phase, trial)."""
    out = {}
    for path in find_summaries(root):
        phase = "training" if "combined_all_training_" in os.path.basename(path) else "experiment"
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for r in data:
            name = match_distribution({"claw_mu": r.get("force_mu"), "claw_sigma": r.get("force_sigma")})
            if name is None or "ir_success" not in r:
                continue
            out[(r.get("participant_id"), phase, r.get("trial_number"))] = (
                name, _true(r.get("ir_success")), _num(r.get("grip_force")), phase)
    return list(out.values())


def load_test_rows(root):
    """Automatic grip tests (grip_offset_test_*.csv / .json).
    Position = MEASURED grab position if it was saved, otherwise the TARGET
    the claw was sent to (ball + requested offset). Same trial in .csv and
    .json is counted once."""
    files = []
    for ext in ("csv", "json"):
        files += glob.glob(os.path.join(root, "**", f"grip_offset_test_*.{ext}"), recursive=True)
    rows, n_meas, n_target = {}, 0, 0
    for path in sorted(set(files)):
        try:
            if path.endswith(".csv"):
                with open(path, newline="") as f:
                    data = list(csv.DictReader(f))
            else:
                with open(path) as f:
                    data = json.load(f)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        for r in data:
            dist_name = match_distribution(r)
            if ONLY_LISTED_DISTRIBUTIONS and dist_name is None:
                continue
            gx, gy = _num(r.get("grab_measured_x")), _num(r.get("grab_measured_y"))
            src = "measured"
            if gx is None or gy is None:
                gx, gy = _num(r.get("target_x")), _num(r.get("target_y"))
                src = "target"
            if gx is None or gy is None:
                continue
            key = r.get("timestamp") or (path, r.get("trial_number"))
            if key in rows and rows[key]["pos_source"] == "measured":
                continue
            rows[key] = {
                "participant_id": "grip test", "phase": "test",
                "trial_number": int(_num(r.get("trial_number")) or 0), "grip_type": r.get("claw") or "",
                "grab_x": gx, "grab_y": gy, "hit": _true(r.get("success_ir")),
                "pos_source": src, "distribution": dist_name, "force": _num(r.get("force")),
            }
    rows = list(rows.values())
    n_meas = sum(r["pos_source"] == "measured" for r in rows)
    if rows:
        print(f"{len(rows)} test trials: {n_meas} with measured position, "
              f"{len(rows) - n_meas} using the target position (no measured position saved)")
    return rows


def assign_goals(rows):
    for r in rows:
        best = min(GOALS.items(),
                   key=lambda g: math.hypot(r["grab_x"] - g[1][0], r["grab_y"] - g[1][1]))
        name, (gx, gy) = best
        r["dx"] = r["grab_x"] - gx
        r["dy"] = r["grab_y"] - gy
        r["distance"] = math.hypot(r["dx"], r["dy"])
        r["nearest_goal"] = name if r["distance"] <= MAX_DIST else "off target"
    return rows


# ---------------- aim area (ellipse) ----------------
def aim_areas(rows, coverage=AREA_COVERAGE):
    """Per goal: mean + covariance of the grab offsets (dx, dy) -> ellipse holding
    `coverage` of the grabs (2-D Gaussian). A point (dx, dy) is inside when
    (d-mean) . inv(cov) . (d-mean) <= r2."""
    r2 = -2 * math.log(1 - coverage)
    areas = {}
    for name, (x0, y0) in GOALS.items():
        pts = np.array([(r["dx"], r["dy"]) for r in rows
                        if r["nearest_goal"] == name and r["distance"] <= AREA_MAX_DIST])
        if len(pts) < 5:
            continue
        areas[name] = {"goal_x": x0, "goal_y": y0, "n": int(len(pts)),
                       "mean": pts.mean(0).tolist(), "cov": np.cov(pts.T).tolist(),
                       "r2": r2, "coverage": coverage}
    return areas


def draw_area(ax, area, cx=0.0, cy=0.0, label=True):
    from matplotlib.patches import Ellipse
    cov = np.array(area["cov"])
    vals, vecs = np.linalg.eigh(cov)
    k = math.sqrt(area["r2"])
    ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    mx, my = area["mean"]
    ax.add_patch(Ellipse((cx + mx, cy + my), 2 * k * math.sqrt(vals[1]), 2 * k * math.sqrt(vals[0]),
                         angle=ang, fill=False, ec=INK, lw=1.8, ls="--", zorder=8,
                         label=f"aim area ({area['coverage']:.0%} of grabs)" if label else None))


# ---------------- plotting ----------------
def kde_grid(xs, ys, bw, xlim, ylim):
    gx = np.linspace(*xlim, 400)
    gy = np.linspace(*ylim, 360)
    X, Y = np.meshgrid(gx, gy)
    Z = np.zeros_like(X)
    for x, y in zip(xs, ys):
        Z += np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * bw ** 2))
    return gx, gy, Z


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def draw_goal(ax, x, y, label=None, label_left=False):
    ax.scatter([x], [y], s=260, marker="o", facecolor="none",
               edgecolor=INK, linewidth=2, zorder=6)
    ax.scatter([x], [y], s=30, marker="+", color=INK, linewidth=1.5, zorder=6)
    if label:
        ax.annotate(label, (x, y), xytext=(-14 if label_left else 14, 30),
                    textcoords="offset points", ha="right" if label_left else "left",
                    fontsize=10, color=INK, weight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def draw_points(ax, rows, xkey, ykey, small=False):
    """small=True: thin, see-through markers so the heat map stays visible."""
    hits = [r for r in rows if r["hit"]]
    miss = [r for r in rows if not r["hit"]]
    k = 0.45 if small else 1.0
    alpha = 0.75 if small else 1.0
    if hits:
        ax.scatter([r[xkey] for r in hits], [r[ykey] for r in hits], s=28 * k,
                   marker="o", facecolor="none", edgecolor=INK, linewidth=1.2 * k + 0.3,
                   alpha=alpha, zorder=4)
    if miss:
        ax.scatter([r[xkey] for r in miss], [r[ykey] for r in miss], s=34 * k,
                   marker="x", color=MISS, linewidth=1.6 * k + 0.3, alpha=alpha, zorder=5)


def plot(rows, title, out_png, areas=None):
    areas = areas or {}
    on = [r for r in rows if r["nearest_goal"] != "off target"]
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.3, 1], hspace=0.55, wspace=0.12)

    # --- arena ---
    ax = fig.add_subplot(gs[:, 0])
    style(ax)
    xs = [r["grab_x"] for r in rows]
    ys = [r["grab_y"] for r in rows]
    xlim = (min(ARENA_X[0], min(xs) - 20), max(ARENA_X[1], max(xs) + 20))
    ylim = (min(ARENA_Y[0], min(ys) - 20), max(ARENA_Y[1], max(ys) + 20))
    gx, gy, Z = kde_grid(xs, ys, KDE_BANDWIDTH, xlim, ylim)
    im = ax.pcolormesh(gx, gy, Z / Z.max(), cmap=HEAT, norm=PowerNorm(FADE, 0, 1),
                       shading="auto", zorder=1)
    draw_points(ax, rows, "grab_x", "grab_y", small=SMALL_POINTS)
    for name, (x, y) in GOALS.items():
        draw_goal(ax, x, y, name, label_left=(x > xlim[1] - 120))
        if name in areas:
            draw_area(ax, areas[name], x, y, label=False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x", color=MUTED)
    ax.set_ylabel("y", color=MUTED)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
    cb.set_label("grab density (relative)", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    n_hit = sum(r["hit"] for r in rows)
    ax.set_title(f"Where the claw went down  ·  {len(rows)} grabs, "
                 f"{n_hit} hits, {len(rows) - n_hit} misses",
                 loc="left", fontsize=12, color=INK)
    legend = [
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor=INK, markersize=7, label="hit (ball in box)"),
        Line2D([], [], marker="x", ls="", color=MISS, markersize=8,
               markeredgewidth=1.6, label="miss"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=INK, markersize=12, markeredgewidth=2, label="ball position"),
    ]
    if areas:
        cov_ = next(iter(areas.values()))["coverage"]
        legend.append(Line2D([], [], color=INK, ls="--", lw=1.8,
                             label=f"aim area ({cov_:.0%} of grabs)"))
    ax.legend(handles=legend, frameon=True, framealpha=0.9, edgecolor="none",
              fontsize=9, loc="upper left", labelcolor=INK)

    # --- one zoom per goal: offsets from the goal ---
    for i, (name, (x0, y0)) in enumerate(GOALS.items()):
        a = fig.add_subplot(gs[i, 1])
        style(a)
        sub = [r for r in on if r["nearest_goal"] == name]
        a.axhline(0, color=GRID, lw=1, zorder=0)
        a.axvline(0, color=GRID, lw=1, zorder=0)
        for rad in (20, 40, 60):
            a.add_patch(plt.Circle((0, 0), rad, fill=False, color=GRID, lw=1, zorder=0))
        if sub:
            dxs = [r["dx"] for r in sub]
            dys = [r["dy"] for r in sub]
            if len(sub) > 1:
                g1 = np.linspace(-ZOOM, ZOOM, 200)
                Xg, Yg = np.meshgrid(g1, g1)
                Zg = np.zeros_like(Xg)
                for dx, dy in zip(dxs, dys):
                    Zg += np.exp(-((Xg - dx) ** 2 + (Yg - dy) ** 2) / (2 * ZOOM_BANDWIDTH ** 2))
                a.pcolormesh(g1, g1, Zg / Zg.max(), cmap=HEAT, norm=PowerNorm(FADE, 0, 1),
                             shading="auto", zorder=0, alpha=0.9)
            draw_points(a, sub, "dx", "dy", small=SMALL_POINTS)
            mx, my = float(np.mean(dxs)), float(np.mean(dys))
            a.scatter([mx], [my], s=90, marker="D", color=INK, edgecolor="white",
                      linewidth=1.2, zorder=7)
            a.annotate("mean", (mx, my), xytext=(8, 6), textcoords="offset points",
                       fontsize=8, color=INK,
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            hit_rate = 100 * sum(r["hit"] for r in sub) / len(sub)
            sub_title = (f"{name} ({x0:.0f}, {y0:.0f})\n"
                         f"n={len(sub)}  ·  hit {hit_rate:.0f}%  ·  "
                         f"mean offset ({mx:+.0f}, {my:+.0f})")
        else:
            sub_title = f"{name} ({x0:.0f}, {y0:.0f})\nno grabs"
        draw_goal(a, 0, 0)
        if name in areas:
            draw_area(a, areas[name], label=False)
        a.set_xlim(-ZOOM, ZOOM)
        a.set_ylim(-ZOOM, ZOOM)
        a.set_aspect("equal")
        a.set_title(sub_title, loc="left", fontsize=10, color=INK)
        a.set_ylabel("offset in y", color=MUTED, fontsize=9)
        if i == len(GOALS) - 1:
            a.set_xlabel("offset in x (grab − ball)", color=MUTED, fontsize=9)

    off = len(rows) - len(on)
    note = f"\n{off} grab(s) more than {MAX_DIST:.0f} from every goal: shown in the arena, left out of the goal panels" if off else ""
    fig.suptitle(title + note, x=0.06, y=0.97, ha="left", fontsize=13, color=INK)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")


# ---------------- bar plot per distribution ----------------
def bar_plot(outcomes, title, out_png):
    """outcomes: list of (distribution name, hit). One bar per listed distribution."""
    names, labels, k_hit, n_tot = [], [], [], []
    print("\nsuccess per distribution (all trials):")
    for name, c in grip_distribution_types.items():
        mu, sd = c["force"]["mu"], c["force"]["sigma"]
        hits = [o[1] for o in outcomes if o[0] == name]
        k = sum(hits)
        names.append(name)
        labels.append(f"{name.replace('_', ' ').title()}\n(μ={mu:g}, σ={sd:g})")
        k_hit.append(k)
        n_tot.append(len(hits))
        print(f"  {name:12s} {k}/{len(hits)}" + (f" = {k/len(hits):.1%}" if hits else "  no trials"))
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
    ax.grid(axis="y", color=GRID, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{title} · {sum(n_tot)} trials", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")


def bar_plot_phases(outcomes, title, out_png):
    """outcomes: (distribution, hit, force, phase). Two bars per distribution:
    training and experiment, side by side."""
    phases = [("training", "Training", "#9ec5f4"), ("experiment", "Experiment", "#256abf")]
    names = list(grip_distribution_types)
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 7.5))
    print("\nsuccess per distribution, training vs experiment:")
    totals = {}
    for j, (ph, label, col) in enumerate(phases):
        rates, ks, ns = [], [], []
        for name in names:
            hits = [o[1] for o in outcomes if o[0] == name and o[3] == ph]
            ks.append(sum(hits)); ns.append(len(hits))
            rates.append(100 * sum(hits) / len(hits) if hits else 0)
        totals[ph] = sum(ns)
        pos = x + (j - 0.5) * width
        ax.bar(pos, rates, width=width * 0.95, color=col, edgecolor="black", linewidth=1.2,
               label=f"{label} ({sum(ns)} trials)", zorder=3)
        for p, r, k, n in zip(pos, rates, ks, ns):
            ax.text(p, r + 1.5, f"{r:.0f}%\n({k}/{n})" if n else "no\ntrials",
                    ha="center", va="bottom", fontsize=10, weight="bold")
        print(f"  {label:10s} " + "  ".join(f"{nm} {k}/{n}" for nm, k, n in zip(names, ks, ns)))
    labels = [f"{n.replace('_', ' ').title()}\n(μ={grip_distribution_types[n]['force']['mu']:g}, "
              f"σ={grip_distribution_types[n]['force']['sigma']:g})" for n in names]
    ax.set_xticks(x, labels, fontsize=12, weight="bold")
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_ylabel("Success Rate", fontsize=14, weight="bold")
    ax.grid(axis="y", color=GRID, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ax.set_title(title, fontsize=14, weight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")


# ---------------- drawn force vs success ----------------
DIST_COLORS = ["#2a78d6", "#1baf8f", "#eb6834", "#a855c7", "#d4a017", "#6b6a64"]
DIST_MARKERS = ["o", "s", "^", "D", "P", "X"]


def force_success_plot(outcomes, title, out_png, seed=0):
    """outcomes: (distribution, hit, force). One dot per trial at its drawn force,
    pass at the top, fail at the bottom (slightly jittered so dots don't overlap);
    dashed: each distribution's Gaussian N(mu, sigma), all scaled to the same height."""
    pts = [o for o in outcomes if o[2] is not None]
    if not pts:
        print("[force plot] no trials with a drawn force - skipped")
        return
    rng = np.random.default_rng(seed)
    forces = [o[2] for o in pts]
    lo = min(min(forces), min(c["force"]["mu"] - 3 * c["force"]["sigma"] for c in grip_distribution_types.values())) - 1
    hi = max(max(forces), max(c["force"]["mu"] + 3 * c["force"]["sigma"] for c in grip_distribution_types.values())) + 1
    xs = np.linspace(lo, hi, 500)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    for i, (name, c) in enumerate(grip_distribution_types.items()):
        col, mk = DIST_COLORS[i % len(DIST_COLORS)], DIST_MARKERS[i % len(DIST_MARKERS)]
        mu, sd = c["force"]["mu"], c["force"]["sigma"]
        # Gaussian shape, peak at 0.9 above the "pass" row
        g = np.exp(-0.5 * ((xs - mu) / sd) ** 2)
        ax.plot(xs, 1.12 + 0.75 * g, ls="--", lw=1.6, color=col, alpha=0.85)
        sub = [o for o in pts if o[0] == name]
        if not sub:
            continue
        f = np.array([o[2] for o in sub])
        y = np.array([1.0 if o[1] else 0.0 for o in sub]) + rng.uniform(-0.06, 0.06, len(sub))
        k = sum(o[1] for o in sub)
        ax.scatter(f, y, s=34, marker=mk, color=col, edgecolor="white", lw=0.6, alpha=0.9,
                   label=f"{name.replace('_', ' ')} (μ={mu:g}, σ={sd:g})  ·  {k}/{len(sub)} = {k/len(sub):.0%}",
                   zorder=3)
    ax.set_yticks([0, 1], ["Fail (0)", "Pass (1)"])
    ax.set_ylim(-0.3, 2.0)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("actual drawn force", color=MUTED)
    ax.set_ylabel("IR success", color=MUTED)
    ax.grid(axis="x", color=GRID, ls="--", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title(f"{title} · {len(pts)} trials   (dashed = each distribution's Gaussian)",
                 loc="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")


def _force_panel(ax, pts, xs, lo, hi, rng, title):
    for i, (name, c) in enumerate(grip_distribution_types.items()):
        col, mk = DIST_COLORS[i % len(DIST_COLORS)], DIST_MARKERS[i % len(DIST_MARKERS)]
        mu, sd = c["force"]["mu"], c["force"]["sigma"]
        g = np.exp(-0.5 * ((xs - mu) / sd) ** 2)
        ax.plot(xs, 1.12 + 0.75 * g, ls="--", lw=1.6, color=col, alpha=0.85)
        sub = [o for o in pts if o[0] == name]
        lab = f"{name.replace('_', ' ')} (μ={mu:g}, σ={sd:g})  ·  "
        if sub:
            f = np.array([o[2] for o in sub])
            y = np.array([1.0 if o[1] else 0.0 for o in sub]) + rng.uniform(-0.06, 0.06, len(sub))
            k = sum(o[1] for o in sub)
            ax.scatter(f, y, s=34, marker=mk, color=col, edgecolor="white", lw=0.6, alpha=0.9,
                       label=lab + f"{k}/{len(sub)} = {k/len(sub):.0%}", zorder=3)
        else:
            ax.scatter([], [], s=34, marker=mk, color=col, label=lab + "no trials")
    ax.set_yticks([0, 1], ["Fail (0)", "Pass (1)"])
    ax.set_ylim(-0.3, 2.0)
    ax.set_xlim(lo, hi)
    ax.set_ylabel("IR success", color=MUTED)
    ax.grid(axis="x", color=GRID, ls="--", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title(title, loc="left", fontsize=12, color=INK)


def force_success_phases(outcomes, title, out_png, seed=0):
    """outcomes: (distribution, hit, force, phase). The force plot with all distributions
    together, one panel for training and one for experiment (same x axis)."""
    pts = [o for o in outcomes if o[2] is not None and len(o) > 3]
    if not pts:
        print("[force plot phases] no trials with a drawn force - skipped")
        return
    rng = np.random.default_rng(seed)
    forces = [o[2] for o in pts]
    dists = grip_distribution_types.values()
    lo = min(min(forces), min(c["force"]["mu"] - 3 * c["force"]["sigma"] for c in dists)) - 1
    hi = max(max(forces), max(c["force"]["mu"] + 3 * c["force"]["sigma"] for c in dists)) + 1
    xs = np.linspace(lo, hi, 500)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for ax, phase in zip(axes, ("training", "experiment")):
        sub = [o for o in pts if o[3] == phase]
        k = sum(o[1] for o in sub)
        info = f"{k}/{len(sub)} = {k/len(sub):.0%}" if sub else "no trials"
        _force_panel(ax, sub, xs, lo, hi, rng, f"{phase.capitalize()} · {len(sub)} trials · {info}")
        print(f"  force plot {phase:10s} {len(sub)} trials")
    axes[-1].set_xlabel("actual drawn force", color=MUTED)
    fig.suptitle(f"{title}   (dashed = each distribution's Gaussian)", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")


# ---------------- tables ----------------
def write_tables(rows, out_dir):
    cols = ["participant_id", "phase", "trial_number", "grip_type", "grab_x",
            "grab_y", "nearest_goal", "dx", "dy", "distance", "hit"]
    with open(os.path.join(out_dir, "goal_errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["participant_id"], r["phase"], r["trial_number"] or 0)):
            w.writerow(r)

    with open(os.path.join(out_dir, "goal_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "goal", "n", "hits", "hit_rate", "mean_dx",
                    "mean_dy", "mean_distance", "median_distance"])
        for phase in ("all", "training", "experiment"):
            for goal in list(GOALS) + ["off target"]:
                sub = [r for r in rows if r["nearest_goal"] == goal
                       and (phase == "all" or r["phase"] == phase)]
                if not sub:
                    continue
                d = [r["distance"] for r in sub]
                w.writerow([phase, goal, len(sub), sum(r["hit"] for r in sub),
                            round(sum(r["hit"] for r in sub) / len(sub), 3),
                            round(float(np.mean([r["dx"] for r in sub])), 1),
                            round(float(np.mean([r["dy"] for r in sub])), 1),
                            round(float(np.mean(d)), 1), round(float(np.median(d)), 1)])
    print(f"saved goal_errors.csv and goal_summary.csv in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="claw_data folder (searched recursively)")
    ap.add_argument("--out", default=None, help="output folder (default: the data folder)")
    ap.add_argument("--coverage", type=float, default=AREA_COVERAGE,
                    help="aim area holds this share of the grabs (default 0.8)")
    ap.add_argument("--tests", action="store_true",
                    help="plot the automatic grip tests (grip_offset_test_*) instead of participants")
    args = ap.parse_args()
    out_dir = args.out or args.folder
    os.makedirs(out_dir, exist_ok=True)

    if args.tests:
        rows = assign_goals(load_test_rows(args.folder))
        if not rows:
            print(f"No grip_offset_test_* trials in {args.folder}" +
                  (" with the distributions listed in grip_distribution_types"
                   if ONLY_LISTED_DISTRIBUTIONS else ""))
            return
        areas = None
        af = AIM_AREAS_FILE or os.path.join(args.folder, "aim_areas.json")
        if os.path.isfile(af):
            with open(af) as f:
                areas = json.load(f)
            print(f"drawing aim areas from {af}")
        if ONLY_LISTED_DISTRIBUTIONS:
            for name in grip_distribution_types:
                n = sum(r["distribution"] == name for r in rows)
                print(f"  {name:12s} {n} trials")
        plot(rows, f"Automatic grip tests · {len(rows)} trials"
             + (" · listed distributions only" if ONLY_LISTED_DISTRIBUTIONS else ""),
             os.path.join(out_dir, "goal_heatmap_tests.png"), areas)
        test_out = [(r["distribution"], r["hit"], r.get("force")) for r in rows if r.get("distribution")]
        bar_plot(test_out, "Success Rate by Grip Distribution Type (grip tests)",
                 os.path.join(out_dir, "success_by_distribution_tests.png"))
        force_success_plot(test_out, "Drawn force vs IR success (grip tests)",
                           os.path.join(out_dir, "force_vs_success_tests.png"))
        return

    if not find_summaries(args.folder):
        print(f"No combined_all_training_*.json / combined_all_trials_*.json in {args.folder}.\n"
              f"For the automatic grip tests use:  python goal_heatmap.py {args.folder} --tests")
        return
    rows = assign_goals(load_rows(args.folder))
    if not rows:
        print("No grab positions found in any combined summary.")
        return
    pids = sorted({r["participant_id"] for r in rows})
    print(f"{len(rows)} grabs from {len(pids)} participant(s): {', '.join(pids)}")

    areas = aim_areas(rows, args.coverage)
    area_file = os.path.join(out_dir, "aim_areas.json")
    with open(area_file, "w") as f:
        json.dump(areas, f, indent=2)
    print(f"Aim areas ({args.coverage:.0%} of grabs, from all trials) saved to {area_file}")
    for name, a in areas.items():
        sd = np.sqrt(np.diag(a["cov"]))
        print(f"  {name}: centre offset ({a['mean'][0]:+.1f}, {a['mean'][1]:+.1f}), "
              f"spread x {sd[0]:.1f}, y {sd[1]:.1f}, n={a['n']}")
    plot(rows, f"All trials · {len(pids)} participant(s)",
         os.path.join(out_dir, "goal_heatmap_all.png"), areas)
    for phase in ("training", "experiment"):
        sub = [r for r in rows if r["phase"] == phase]
        if sub:
            plot(sub, f"{phase.capitalize()} trials · {len({r['participant_id'] for r in sub})} participant(s)",
                 os.path.join(out_dir, f"goal_heatmap_{phase}.png"))
    write_tables(rows, out_dir)
    part_out = load_participant_outcomes(args.folder)
    bar_plot(part_out, "Success Rate by Grip Distribution Type (participants)",
             os.path.join(out_dir, "success_by_distribution.png"))
    bar_plot_phases(part_out, "Success Rate by Grip Distribution Type: training vs experiment",
                    os.path.join(out_dir, "success_by_distribution_phases.png"))
    force_success_plot(part_out, "Drawn force vs IR success (participants)",
                       os.path.join(out_dir, "force_vs_success.png"))
    force_success_phases(part_out, "Drawn force vs IR success: training vs experiment",
                         os.path.join(out_dir, "force_vs_success_phases.png"))


if __name__ == "__main__":
    main()