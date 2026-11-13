"""Which files have the trials around force 196 that almost all fail?
Run next to plot_sigmoid.py (or sigmoidplot.py):   python check_196.py  [folder]
"""
import os
import sys
from collections import defaultdict

try:
    import plot_sigmoid as ps
except ImportError:
    import sigmoidplot as ps

LO, HI = 195.5, 196.7
folder = sys.argv[1] if len(sys.argv) > 1 else ps.DEFAULT_FOLDER
per_file = defaultdict(lambda: [0, 0, set()])
seen = set()
for root, _, names in os.walk(folder):
    for name in sorted(names):
        if not name.endswith((".csv", ".json")):
            continue
        path = os.path.join(root, name)
        try:
            recs = ps._records(path)
        except Exception:
            continue
        for r in recs:
            p = ps._parse(r, path)
            if not p or p[0] in seen:
                continue
            seen.add(p[0])
            key, force, off, ok = p
            if LO <= force < HI:
                d = per_file[os.path.relpath(path, folder)]
                d[0] += 1
                d[1] += ok
                d[2].add(round(force, 2))
print(f"trials with force {LO}-{HI}, per file:")
for f, (n, k, forces) in sorted(per_file.items(), key=lambda x: -x[1][0]):
    fs = sorted(forces)
    print(f"  {n:5d} trials  {k:4d} success ({k/n:4.0%})  forces {fs[0]}-{fs[-1]}"
          f"{'  (all the same force!)' if len(fs) == 1 else ''}   {f}")
