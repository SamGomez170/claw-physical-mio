"""
Plot the psychometric sigmoid P(success) vs force from every .csv AND .json file
in a folder (default /home/clawMachine/claw_data/offset_data).

    python plot_sigmoid.py
    python plot_sigmoid.py --offsets 0              # only aimed at the ball
    python plot_sigmoid.py --offsets 0 10 --threshold 0.8
    python plot_sigmoid.py /some/other/folder
    python plot_sigmoid.py --aim-area              # only test trials INSIDE the participants'
                                                   # aim area (dashed ellipse of goal_heatmap.py)
    python plot_sigmoid.py /new/tests --aim-area --areas /participants/aim_areas.json
    python plot_sigmoid.py --aim-area --weighted   # inside the area, and trials near the
                                                   # CENTRE of the ellipse count more

Understands:
  - grip test files (CSV or JSON): force, success_ir, offset_radius
  - game files (combined_all_*.json etc.): grip_force, ir_success, grab_x/grab_y
    -> offset = distance from the grab to the nearest ball, rounded to 10
The same trial saved in both a .csv and a .json is counted only once.
Fit:  P = (1 - lapse) / (1 + exp(-(F - a) / b))
      a = force at the middle of the curve, b = slope, lapse = fails even at full force
Output: sigmoid.png in the same folder (or --out).
Needs numpy, scipy, matplotlib.
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_FOLDER = "/home/clawMachine/claw_data/offset_data"
BALLS = [(445, 385), (625, 150), (750, 450)]


def _true(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


GOALS = {"Goal 1": (750.0, 450.0), "Goal 2": (625.0, 150.0), "Goal 3": (445.0, 385.0)}
AREA_MAX_DIST = 80.0      # same as goal_heatmap.py


def _nearest_goal(x, y):
    name, (gx, gy) = min(GOALS.items(), key=lambda g: math.hypot(x - g[1][0], y - g[1][1]))
    return name, x - gx, y - gy


def aim_areas(folder, coverage, areas_file=None):
    """Same ellipses as goal_heatmap.py: read aim_areas.json (--areas, or the one in
    the folder), otherwise compute them from the participant files in the folder."""
    if areas_file:
        with open(areas_file) as f:
            areas = json.load(f)
        print(f"aim areas read from {areas_file}")
        return areas
    path = os.path.join(folder, "aim_areas.json")
    if os.path.isfile(path):
        with open(path) as f:
            areas = json.load(f)
        if areas and abs(next(iter(areas.values()))["coverage"] - coverage) < 1e-9:
            print(f"aim areas read from {path}")
            return areas
    # look for an aim_areas.json made by goal_heatmap.py one level up (e.g. ~/claw_data)
    found = sorted(glob.glob(os.path.join(os.path.dirname(folder), "**", "aim_areas.json"),
                             recursive=True), key=os.path.getmtime)
    if found:
        with open(found[-1]) as f:
            areas = json.load(f)
        print(f"aim areas read from {found[-1]} (newest one found)")
        return areas
    grabs = {}
    for p in glob.glob(os.path.join(folder, "**", "combined_all_*.json"), recursive=True):
        phase = "training" if "training" in os.path.basename(p) else "trial"
        try:
            recs = _records(p)
        except Exception:
            continue
        for r in recs:
            gx, gy = _num(r.get("grab_x")), _num(r.get("grab_y"))
            if gx is not None and gy is not None:
                grabs[(r.get("participant_id"), phase, r.get("trial_number"))] = (gx, gy)
    r2 = -2 * math.log(1 - coverage)
    areas = {}
    for name in GOALS:
        pts = []
        for gx, gy in grabs.values():
            g, dx, dy = _nearest_goal(gx, gy)
            if g == name and math.hypot(dx, dy) <= AREA_MAX_DIST:
                pts.append((dx, dy))
        if len(pts) >= 5:
            pts = np.array(pts)
            areas[name] = {"n": len(pts), "mean": pts.mean(0).tolist(),
                           "cov": np.cov(pts.T).tolist(), "r2": r2, "coverage": coverage}
    if grabs:
        print(f"aim areas computed from {len(grabs)} participant grabs")
    return areas


def area_dist2(r, areas):
    """Squared ellipse distance of a test trial from the centre of its goal's aim area
    (0 = centre, r2 = on the dashed line). None if it cannot be computed.
    Uses the measured grab position, or the requested offset when none was saved."""
    bx, by = _num(r.get("ball_x")), _num(r.get("ball_y"))
    if bx is None or by is None:
        return None
    mx, my = _num(r.get("grab_measured_x")), _num(r.get("grab_measured_y"))
    if mx is not None and my is not None:
        dx, dy = mx - bx, my - by
    else:
        dx, dy = _num(r.get("offset_dx")) or 0.0, _num(r.get("offset_dy")) or 0.0
    name, _, _ = _nearest_goal(bx, by)
    a = areas.get(name)
    if a is None:
        return None
    d = np.array([dx, dy]) - np.array(a["mean"])
    return float(d @ np.linalg.inv(np.array(a["cov"])) @ d), a["r2"]


def in_area(r, areas):
    res = area_dist2(r, areas)
    return res is not None and res[0] <= res[1]


def area_weight(r, areas):
    """Weight from how close the trial is to the centre of the ellipse:
    1 at the centre, falling like the participants' own aim distribution
    (about 0.2 on the dashed line for an 80 % area)."""
    res = area_dist2(r, areas)
    return 0.0 if res is None else math.exp(-res[0] / 2)


def _parse(r, source):
    """One record -> (key, force, offset, success) or None."""
    force = _num(r.get("force"))
    if force is not None and ("success_ir" in r or "success_combined" in r):     # grip test
        off = _num(r.get("offset_radius")) or 0.0
        ok = _true(r.get("success_ir", r.get("success_combined")))
        key = ("test", r.get("timestamp") or (source, r.get("trial_number")))
        return key, force, off, ok
    force = _num(r.get("grip_force"))
    if force is not None and "ir_success" in r:                                   # game trial
        gx, gy = _num(r.get("grab_x")), _num(r.get("grab_y"))
        if gx is None or gy is None:
            off = -1.0                                                            # unknown aim
        else:
            d = min(math.hypot(gx - bx, gy - by) for bx, by in BALLS)
            off = float(round(d / 10) * 10)
        phase = "training" if "training" in os.path.basename(source) else "trial"
        key = ("game", r.get("participant_id"), phase, r.get("trial_number"))
        return key, force, off, _true(r.get("ir_success"))
    return None


def _records(path):
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):                     # {"trials": [...]} or a single record
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return [data]
    return [r for r in data if isinstance(r, dict)]


def load(folder, offsets, no_game=False, areas=None, weighted=False):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True) +
                   glob.glob(os.path.join(folder, "**", "*.json"), recursive=True))
    trials, used = {}, []
    for path in files:
        try:
            recs = _records(path)
        except Exception as e:
            print(f"  skipped {os.path.basename(path)}: {e}")
            continue
        n0 = len(trials)
        for r in recs:
            p = _parse(r, path)
            if p and no_game and p[0][0] == "game":
                continue
            if p and areas is not None and (p[0][0] != "test" or not in_area(r, areas)):
                continue
            if p:
                w = area_weight(r, areas) if weighted else 1.0
                trials[p[0]] = p[1:] + (w,)
        if len(trials) > n0:
            used.append(path)
    F, Y, W = [], [], []
    for force, off, ok, w in trials.values():
        if offsets is not None and off not in offsets:
            continue
        F.append(force)
        Y.append(ok)
        W.append(w)
    return used, np.array(F), np.array(Y, float), np.array(W)


def sigmoid(F, a, b, lapse):
    return (1 - lapse) / (1 + np.exp(-(F - a) / b))


def fit(F, Y, W=None):
    W = np.ones_like(F) if W is None else W
    def nll(p):
        a, logb, lapse = p
        q = np.clip(sigmoid(F, a, np.exp(logb), lapse), 1e-6, 1 - 1e-6)
        return -np.sum(W * (Y * np.log(q) + (1 - Y) * np.log(1 - q)))
    best = None
    for a0 in np.arange(F.min(), F.max() + 1, 2.0):        # several starts, keep the best
        r = minimize(nll, [a0, 0.5, 0.05], method="L-BFGS-B",
                     bounds=[(F.min() - 10, F.max() + 10), (np.log(0.2), np.log(10)), (0.0, 0.6)])
        if best is None or r.fun < best.fun:
            best = r
    a, logb, lapse = best.x
    return a, np.exp(logb), lapse


def force_at(p, a, b, lapse):
    """Force where the curve reaches p (None if it never gets there)."""
    if p >= 1 - lapse:
        return None
    return a + b * np.log(p / (1 - lapse - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=DEFAULT_FOLDER)
    ap.add_argument("--offsets", type=float, nargs="+", default=None,
                    help="only use these requested offsets, e.g. --offsets 0 10")
    ap.add_argument("--no-game", action="store_true",
                    help="ignore participant game files (combined_all_*.json), use only grip tests")
    ap.add_argument("--aim-area", action="store_true",
                    help="only grip-test trials inside the participants' aim area (ellipse)")
    ap.add_argument("--weighted", action="store_true",
                    help="with --aim-area: trials near the centre of the ellipse count more")
    ap.add_argument("--areas", default=None,
                    help="aim_areas.json made by goal_heatmap.py (e.g. from the participant folder)")
    ap.add_argument("--coverage", type=float, default=0.8,
                    help="aim area holds this share of participant grabs (default 0.8)")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--bin", type=float, default=1.0, help="bin width for the dots")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    folder = os.path.abspath(os.path.expanduser(args.folder))
    if not os.path.isdir(folder):
        parent = os.path.dirname(folder)
        msg = f"Folder does not exist: {folder}"
        if os.path.isdir(parent):
            msg += f"\nFolders/files in {parent}:\n  " + "\n  ".join(sorted(os.listdir(parent)))
        raise SystemExit(msg)
    args.folder = folder
    if args.weighted and not args.aim_area:
        raise SystemExit("--weighted needs --aim-area")
    areas = None
    if args.aim_area:
        areas = aim_areas(folder, args.coverage,
                          os.path.abspath(os.path.expanduser(args.areas)) if args.areas else None)
        if not areas:
            raise SystemExit("--aim-area: no aim_areas.json and no participant files in " + folder +
                             "\nRun goal_heatmap.py on the participant folder first and pass "
                             "--areas <that folder>/aim_areas.json")
        for name, a in areas.items():
            print(f"  {name}: centre offset ({a['mean'][0]:+.1f}, {a['mean'][1]:+.1f}), n={a['n']}")
    files, F, Y, W = load(args.folder, args.offsets, args.no_game, areas, args.weighted)
    if len(F) == 0:
        found = sorted(glob.glob(os.path.join(folder, "**", "*.*"), recursive=True))
        msg = f"No trials found in {folder}\n"
        if not found:
            msg += "The folder is empty."
        else:
            msg += f"{len(found)} files there, but none had trial data:\n  " + \
                   "\n  ".join(os.path.relpath(p, folder) for p in found[:20])
            if args.offsets is not None:
                msg += f"\n(also check --offsets {args.offsets}: maybe no trials at those offsets)"
        raise SystemExit(msg)
    for p in files:
        print(f"  using {os.path.basename(p)}")
    print(f"{len(files)} files, {len(F)} trials (duplicates removed), "
          f"{int(Y.sum())} successes ({Y.mean():.0%})")
    if args.weighted:
        n_eff = W.sum() ** 2 / (W ** 2).sum()
        print(f"weighted by closeness to the ellipse centre: weighted success "
              f"{np.sum(W * Y) / W.sum():.0%}, weights {W.min():.2f}-{W.max():.2f}, "
              f"worth about {n_eff:.0f} unweighted trials")

    a, b, lapse = fit(F, Y, W)
    t = force_at(args.threshold, a, b, lapse)
    print(f"fit: middle a = {a:.2f}, slope b = {b:.2f}, maximum = {1 - lapse:.0%}")
    for p in (0.5, 0.7, 0.8, args.threshold):
        f = force_at(p, a, b, lapse)
        txt = "never reached" if f is None else f"{f:.2f}" + (" (beyond tested forces)" if f > F.max() else "")
        print(f"  {p:.0%} at force {txt}")

    # binned pass rate
    edges = np.arange(np.floor(F.min()), np.ceil(F.max()) + args.bin + 1e-9, args.bin)
    bx, bp, bn = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (F >= lo) & (F < hi)
        if m.any() and W[m].sum() > 0:
            bx.append(np.average(F[m], weights=W[m]))
            bp.append(np.average(Y[m], weights=W[m]))
            bn.append(m.sum())
    bn = np.array(bn)

    xs = np.linspace(np.floor(F.min()) - 1, np.ceil(F.max()) + 1, 400)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(bx, bp, s=25 * bn, color="#3987e5", alpha=0.55, edgecolor="white",
               label="binned pass rate (size = n)", zorder=3)
    for x, p, n in zip(bx, bp, bn):
        ax.text(x, p + 0.04, f"n={n}", ha="center", fontsize=8, color="#6b6a64")
    ax.plot(xs, sigmoid(xs, a, b, lapse), color="#0d366b", lw=2.6,
            label="logistic fit" + (" (weighted)" if args.weighted else ""), zorder=4)
    ax.axhline(args.threshold, color="#6b6a64", lw=0.9, ls="--")
    if t is not None and t <= F.max():
        ax.axvline(t, color="#eb6834", lw=1.2, ls=":")
        ax.annotate(f"{args.threshold:.0%} threshold ≈ {t:.2f}", xy=(t, args.threshold),
                    xytext=(t + 1.0, args.threshold - 0.3), fontsize=11,
                    arrowprops=dict(arrowstyle="->"))
    else:
        ax.text(xs[-1], args.threshold + 0.015,
                f"{args.threshold:.0%} not reached up to force {F.max():.0f}",
                ha="right", color="#eb6834")

    which = "all offsets" if args.offsets is None else "offsets " + ", ".join(f"{o:g}" for o in args.offsets)
    if args.aim_area:
        which = f"tests inside participant aim area ({args.coverage:.0%})" + \
                (", weighted by closeness to centre" if args.weighted else "") + \
                ("" if args.offsets is None else ", " + which)
    rate = np.sum(W * Y) / W.sum()
    ax.set_title(f"P(success) vs force  ·  {len(F)} trials, {which} ({rate:.0%} success)", loc="left")
    ax.set_xlabel("actual force")
    ax.set_ylabel("P(success)")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlim(xs[0], xs[-1])
    ax.grid(color="#e4e3de")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right")

    out = args.out or os.path.join(args.folder, ("sigmoid_aim_area_weighted.png" if args.weighted else "sigmoid_aim_area.png")
                                         if args.aim_area else "sigmoid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()