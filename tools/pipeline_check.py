"""Step B3: the one check of the pipeline on the public five minutes. PASS / FAIL lines only.

    python tools/pipeline_check.py CSV

The only reader of the public GT. Calls the official possession_metrics.py; prints no number, writes
nothing. --gt exists for the test on a synthetic file.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("MTAH_OFFICIAL_REPO") or Path(os.environ["MTAH_DATA_DIR"]) / "more-than-a-hack-2027")


def load_pm():
    spec = importlib.util.spec_from_file_location("pm", REPO / "evaluation" / "possession_metrics.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    return pm


def verdict(name, ok):
    print(name, "PASS" if ok else "FAIL", flush=True)


def macro(pm, g, p, keep):
    return pm.macro_f1(pm.scored_confusion(pm.confusion(g[keep], p[keep])))


def main():
    args = sys.argv[1:]
    gt_path = REPO / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    if "--gt" in args:
        gt_path = args[args.index("--gt") + 1]
    pm = load_pm()
    table = pm.read_label_csv(args[0])
    g = pm.label_codes(pm.read_label_csv(gt_path))
    # K1
    p = pm.label_codes(table)
    runs = pm.valid_runs(g)
    keep = np.zeros(len(g), dtype=bool)
    for a, b in runs:
        keep[a:b] = True
    verdict("K1", pm.validate_submission(table, len(g)) is None and not (p[keep] == pm.X).any())
    # K2
    scores = []
    for d in range(-250, 251):
        q = np.roll(p, d)
        scores.append((macro(pm, g, q, keep), -abs(d), d))
    verdict("K2", abs(max(scores)[2]) <= 25)
    # K3
    q = p.copy()
    q[p == pm.H], q[p == pm.A] = pm.A, pm.H
    verdict("K3", macro(pm, g, p, keep) > macro(pm, g, q, keep))
    # K4
    gd, pd_ = g[keep] == pm.D, p[keep] == pm.D
    k4 = bool(pd_.any() and gd.any() and gd[pd_].mean() > gd.mean() and pd_[gd].mean() > pd_.mean())
    verdict("K4", k4)
    # K5
    lab = set(p[keep].tolist())
    filled = pm.fill_abstentions(p, runs)
    verdict("K5", {pm.H, pm.A, pm.D} <= lab and pm.spell_scores(g, filled, runs)["mean_f1"] > 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
