"""
What would these claws do, judged from the trials you already ran?

It takes EVERY trial in the data folder (grip tests .csv/.json and participant
files combined_all_*.json), whatever claw settings they were run with, and uses
only two things from each trial: the force that was actually used, and whether
the ball landed in the box.

For each claw in grip_distribution_types (its mu and sigma):
  1. the trials are sorted into force bins (BIN_WIDTH wide), and the success
     rate in each bin is counted:  successes / trials  (no curve is fitted)
  2. each bin is weighted by how often that claw would draw a force in it
     (the claw's Gaussian)
  -> estimated success rate of the claw.
"Covered" says how much of the claw's distribution falls on forces you have
data for; bins with fewer than MIN_TRIALS_PER_BIN trials are not used.

It also saves, for each claw, a set of REAL trials picked so that their forces
follow the claw's distribution (resampled_trials.csv).

>>> Edit the settings below, then run:   python claws_from_data.py
Needs numpy and matplotlib.
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

DATA_FOLDER = "/home/clawMachine/claw_data/offset_data"
OUTPUT_FOLDER = None         # None = same as DATA_FOLDER

SOURCE = "both"              # "tests" = grip tests only, "game" = participants only, "both"
MAX_TEST_OFFSET = None       # grip tests: only trials aimed this close to the ball,
                             # e.g. 20 (None = all offsets)

BIN_WIDTH = 0.5              # width of the force bins
MIN_TRIALS_PER_BIN = 5       # bins with fewer trials are treated as "no data"

SAMPLES_PER_CLAW = 50        # real trials per claw written to resampled_trials.csv
RANDOM_SEED = 0
# =====================================================================

COLORS = ["#2a78d6", "#1baf8f", "#eb6834", "#a855c7", "#d4a017", "#6b6a64"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


# ---------------------------------------------------------------- loading
def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _true(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _records(path):
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return [data]
    return [r for r in data if isinstance(r, dict)]


def load(folder):
    """One entry per trial: force actually used + success. The same trial saved
    in a .csv and a .json (or in a rebuilt copy) is counted once."""
    files = sorted(set(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True) +
                       glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)))
    trials = {}
    for path in files:
        name = os.path.basename(path)
        if name == "resampled_trials.csv":
            continue
        try:
            recs = _records(path)
        except Exception as e:
            print(f"  skipped {name}: {e}")
            continue
        for r in recs:
            force = _num(r.get("force"))
            if force is not None and "success_ir" in r:                     # grip test
                if SOURCE == "game":
                    continue
                off = _num(r.get("offset_dist"))
                if off is None:
                    off = _num(r.get("offset_radius")) or 0.0
                if MAX_TEST_OFFSET is not None and off > MAX_TEST_OFFSET:
                    continue
                key = ("test", r.get("timestamp") or (path, r.get("trial_number")))
                trials[key] = {"source": "test", "file": name, "trial": r.get("trial_number"),
                               "force": force, "hit": _true(r.get("success_ir")),
                               "orig_mu": _num(r.get("claw_mu")), "orig_sigma": _num(r.get("claw_sigma"))}
                continue
            force = _num(r.get("grip_force"))
            if force is not None and "ir_success" in r:                     # participant trial
                if SOURCE == "tests":
                    continue
                phase = "training" if "training" in name else "experiment"
                key = ("game", r.get("participant_id"), phase, r.get("trial_number"))
                trials[key] = {"source": f"game {phase}", "file": name, "trial": r.get("trial_number"),
                               "force": force, "hit": _true(r.get("ir_success")),
                               "orig_mu": _num(r.get("force_mu")), "orig_sigma": _num(r.get("force_sigma"))}
    return list(trials.values())


# ---------------------------------------------------------------- estimate
def norm_cdf(x, mu, sigma):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def main():
    folder = os.path.abspath(os.path.expanduser(DATA_FOLDER))
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder does not exist: {folder}")
    out_dir = OUTPUT_FOLDER or folder
    os.makedirs(out_dir, exist_ok=True)

    trials = load(folder)
    if not trials:
        raise SystemExit(f"No trials found in {folder} (SOURCE = {SOURCE!r})")
    F = np.array([t["force"] for t in trials])
    Y = np.array([1.0 if t["hit"] else 0.0 for t in trials])
    n_test = sum(t["source"] == "test" for t in trials)
    print(f"{len(trials)} trials: {n_test} grip tests, {len(trials) - n_test} participant trials"
          f"   forces {F.min():.1f} to {F.max():.1f}, {Y.mean():.0%} success overall")
    if MAX_TEST_OFFSET is not None:
        print(f"grip tests limited to offsets up to {MAX_TEST_OFFSET:g}")

    # force bins: successes / trials in each
    lo = math.floor(F.min() / BIN_WIDTH) * BIN_WIDTH
    hi = math.ceil(F.max() / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH
    edges = np.arange(lo, hi + 1e-9, BIN_WIDTH)
    idx = np.clip(np.digitize(F, edges) - 1, 0, len(edges) - 2)
    n_bin = np.bincount(idx, minlength=len(edges) - 1)
    k_bin = np.bincount(idx, weights=Y, minlength=len(edges) - 1)
    ok = n_bin >= MIN_TRIALS_PER_BIN
    rate = np.where(n_bin > 0, k_bin / np.maximum(n_bin, 1), np.nan)
    mids = (edges[:-1] + edges[1:]) / 2

    rng = np.random.default_rng(RANDOM_SEED)
    by_bin = {b: np.flatnonzero(idx == b) for b in np.flatnonzero(ok)}
    results, picked = [], []
    print(f"\nclaw           mu    sigma   estimated success   covered   trials behind it")
    for name, c in grip_distribution_types.items():
        mu, sd = float(c["force"]["mu"]), float(c["force"]["sigma"])
        # how often this claw draws a force in each bin
        w = np.array([norm_cdf(b, mu, sd) - norm_cdf(a, mu, sd) for a, b in zip(edges[:-1], edges[1:])])
        covered = float(w[ok].sum())
        if covered > 0:
            p = float(np.sum(w[ok] * rate[ok]) / covered)
            # rough uncertainty: each bin's own binomial error, weighted the same way
            var = np.sum((w[ok] / covered) ** 2 * rate[ok] * (1 - rate[ok]) / n_bin[ok])
            se = math.sqrt(var)
            n_used = int(n_bin[ok & (w > 0.005)].sum())
        else:
            p, se, n_used = float("nan"), float("nan"), 0
        results.append({"name": name, "mu": mu, "sigma": sd, "p": p, "se": se,
                        "covered": covered, "n_used": n_used, "w": w})
        txt = "no data" if covered == 0 else f"{p:6.1%}  (+/- {se:.1%})"
        warn = "   <- low coverage, treat with care" if 0 < covered < 0.8 else ""
        print(f"{name:13s} {mu:6g} {sd:6g}   {txt:18s}  {covered:6.0%}    {n_used}{warn}")

        # real trials whose forces follow this claw's distribution
        if covered > 0:
            pb = np.where(ok, w, 0.0)
            pb = pb / pb.sum()
            for b in rng.choice(len(pb), size=SAMPLES_PER_CLAW, p=pb):
                t = trials[int(rng.choice(by_bin[int(b)]))]
                picked.append({"claw": name, "claw_mu": mu, "claw_sigma": sd,
                               "force": round(t["force"], 4), "success": t["hit"],
                               "from": t["source"], "original_mu": t["orig_mu"],
                               "original_sigma": t["orig_sigma"], "file": t["file"],
                               "trial_number": t["trial"]})

    csv_path = os.path.join(out_dir, "resampled_trials.csv")
    if picked:
        with open(csv_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(picked[0]))
            wr.writeheader()
            wr.writerows(picked)
        print(f"\nsaved {csv_path}  ({SAMPLES_PER_CLAW} real trials per claw; a trial can appear more than once)")
        for r in results:
            sub = [p for p in picked if p["claw"] == r["name"]]
            if sub:
                print(f"  {r['name']:13s} {sum(p['success'] for p in sub)}/{len(sub)} successes in this sample")

    # ------------------------------------------------------------ plot
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [1.25, 1]})
    show = n_bin > 0
    ax.scatter(mids[show & ok], rate[show & ok], s=20 + 380 * n_bin[show & ok] / n_bin.max(),
               color="#3987e5", alpha=0.55, edgecolor="white", zorder=3,
               label=f"success in each {BIN_WIDTH:g}-wide force bin (size = number of trials)")
    few = show & ~ok
    if few.any():
        ax.scatter(mids[few], rate[few], s=26, facecolor="none", edgecolor=MUTED, lw=1, zorder=3,
                   label=f"fewer than {MIN_TRIALS_PER_BIN} trials: not used")
    xs = np.linspace(min(edges[0], min(r["mu"] - 3.5 * r["sigma"] for r in results)),
                     max(edges[-1], max(r["mu"] + 3.5 * r["sigma"] for r in results)), 600)
    for i, r in enumerate(results):
        g = np.exp(-0.5 * ((xs - r["mu"]) / r["sigma"]) ** 2)
        ax.plot(xs, 1.08 + 0.3 * g, ls="--", lw=1.6, color=COLORS[i % len(COLORS)],
                label=f"{r['name'].replace('_', ' ')} (μ={r['mu']:g}, σ={r['sigma']:g})")
    ax.axvspan(xs[0], edges[0], color=GRID, alpha=0.5, lw=0)
    ax.axvspan(edges[-1], xs[-1], color=GRID, alpha=0.5, lw=0)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(-0.04, 1.42)
    ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
    ax.set_xlabel("force actually used", color=MUTED)
    ax.set_ylabel("success rate", color=MUTED)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    src = {"both": "grip tests + participants", "tests": "grip tests", "game": "participants"}[SOURCE]
    ax.set_title(f"Your data: {len(trials)} trials ({src})   ·   dashed = the forces each claw would draw"
                 f"   ·   grey = no data", loc="left", fontsize=12, color=INK)

    x = np.arange(len(results))
    vals = [0 if math.isnan(r["p"]) else 100 * r["p"] for r in results]
    errs = [0 if math.isnan(r["se"]) else 100 * r["se"] for r in results]
    ax2.bar(x, vals, width=0.5, color=[COLORS[i % len(COLORS)] for i in range(len(results))],
            edgecolor="black", linewidth=1.3, zorder=3)
    ax2.errorbar(x, vals, yerr=errs, fmt="none", ecolor="black", elinewidth=1.2, capsize=7, zorder=4)
    for i, r in enumerate(results):
        txt = "no data" if r["covered"] == 0 else f"{r['p']:.1%}\ncovered {r['covered']:.0%}"
        ax2.text(x[i], vals[i] + errs[i] + 2, txt, ha="center", va="bottom", fontsize=11, weight="bold")
    ax2.set_xticks(x, [f"{r['name'].replace('_', ' ').title()}\n(μ={r['mu']:g}, σ={r['sigma']:g})"
                       for r in results], fontsize=11, weight="bold")
    ax2.set_ylim(0, 118)
    ax2.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax2.set_ylabel("estimated success rate", color=MUTED)
    ax2.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.set_title("Estimated success of each claw from the data above (bars = rough uncertainty)",
                  loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    png = os.path.join(out_dir, "claws_from_data.png")
    fig.savefig(png, dpi=150, facecolor="white")
    print(f"saved {png}")


if __name__ == "__main__":
    main()