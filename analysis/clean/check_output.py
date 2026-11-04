"""Step B2: checks of section 6 on our CSV. Prints only A<n> PASS / A<n> FAIL lines.

    python -m analysis.clean.check_output CSV REGISTRY.json [--a6] [--blocks JSON --weights DIR]

Reads only the CSV, the ATD, the tags and the XML. A1 and A13 (validator, two runs) and A12 (static
grep, tests) run outside this script; A6 needs the arrays and the weights and runs with --a6.
--blocks and --weights: the fixture only.
"""

import json
import math
import sys

import numpy as np
import pandas as pd

from analysis.clean import bands
from mtah import data, infer
from mtah.transfer import features, ours
from mtah.transfer_mt import net


def verdict(name, ok):
    print(name, "PASS" if ok else "FAIL", flush=True)


def binom_p(k, n):
    """P(X >= k), X ~ Bin(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n


def a4(lab, tags, names, blocks):
    """H <-> A: the majority of live frames under a phase tag agrees with the tag's team."""
    if "Team" not in tags:
        return False
    good, n = 0, 0
    kinds = ("BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES")
    for r in tags.itertuples():
        team = str(getattr(r, "Team")).strip().lower()
        if r.code not in kinds or team not in names:
            continue
        seg = lab[int(r.start_frame):int(r.end_frame)]
        h, a = int((seg == "H").sum()), int((seg == "A").sum())
        if h == a:
            continue
        n += 1
        good += ("H" if h > a else "A") == ("H" if names[team] == 0 else "A")
    return n > 0 and binom_p(good, n) < 0.01


def a5(lab, period, P, slot_team, B, pitch):
    """Time axis: peak of the cross-correlation of our D starts and upward crossings of ball_out_dist."""
    team = np.where(np.isfinite(P[..., 0]), slot_team[None, :], -1).astype(np.int8)
    X, names, _ = features.build(P, team, B, period, pitch, features.ALL_RADII)
    out = X[:, names.index("ball_out_dist")]
    seen = X[:, names.index("ball_present")] > 0.5
    cross = np.zeros(len(lab))
    cross[1:] = (out[1:] > 0) & (out[:-1] <= 0) & seen[1:] & seen[:-1]
    d = np.zeros(len(lab))
    d[1:] = (lab[1:] == "D") & (lab[:-1] != "D") & (lab[:-1] != "X") & (period[1:] > 0)
    ca, da = np.flatnonzero(cross), np.flatnonzero(d)
    best, peak = 0, 0
    for lag in range(-250, 251):
        c = int(np.isin(ca + lag, da).sum())
        if c > best or (c == best and abs(lag) < abs(peak)):
            best, peak = c, lag
    return abs(peak) <= 50


def out_stats(lab, block_list, t0):
    codes = np.where(lab == "H", 0, np.where(lab == "A", 1, np.where(lab == "D", 2, -1))).astype(np.int8)
    return bands.stats(codes, [(h, t0[h], b) for h, _, b in block_list])


def in_band(st, band):
    return (band["poss_per_min"][0] <= st[0] <= band["poss_per_min"][1]
            and band["median_s"][0] <= st[1] <= band["median_s"][1]
            and st[2] <= band["short_share"][1]
            and band["dead_share"][0] <= st[3] <= band["dead_share"][1])


def a10(P, slot_team, B, period, play, pitch, reg, model, limit):
    """Maximum over the three pairs of seeds of the share of frames whose decoded labels differ."""
    weights, cfg = model
    labs = [infer.predict(P, B, slot_team, period, _t0(period, play), pitch, ([w], cfg)) for w in weights]
    ok = period > 0
    worst = 0.0
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            worst = max(worst, float((labs[i] != labs[j])[ok].mean()))
    return worst < limit


def _t0(period, play):
    return {h: int(np.flatnonzero(play == h)[0]) for h in (1, 2) if (play == h).any()}


def a11(P, slot_team, B, play, pitch, reg, cfg):
    team = np.where(np.isfinite(P[..., 0]), slot_team[None, :], -1).astype(np.int8)
    X, names = infer.model_inputs(P, team, B, play, pitch, cfg)
    if list(names) != reg["quantile_names"]:
        return False
    Z = X[play > 0].astype(np.float16).astype(np.float32)
    lo, hi = np.array(reg["quantile_lo"]), np.array(reg["quantile_hi"])
    ok = True
    for k in range(Z.shape[1]):
        x = Z[:, k]
        x = x[np.isfinite(x)]
        if len(x) and float(((x >= lo[k]) & (x <= hi[k])).mean()) < 0.95:
            ok = False
    return ok


def main():
    path, reg_path = sys.argv[1], sys.argv[2]
    reg = json.loads(open(reg_path).read())
    with_a6 = "--a6" in sys.argv
    lab = pd.read_csv(path, dtype={"label": str})["label"].to_numpy().astype(str)
    blocks = ours.blocks()
    weights = infer.WEIGHTS_DIR
    if "--blocks" in sys.argv:
        blocks = [tuple(b) for b in json.loads(sys.argv[sys.argv.index("--blocks") + 1])]
        weights = sys.argv[sys.argv.index("--weights") + 1]
    P, slot_team, B, pitch = ours.match_arrays(blocks)
    tags = data.load_smart_tagging()
    t0 = ours.play_starts(tags, blocks)
    block = ours.block_of(len(B), blocks)
    play = ours.play_period(block, t0, blocks)

    # A2
    verdict("A2", bool(((lab == "X") == (block == 0)).all()))
    # A3
    ok = all((lab[f:t0[h]] == "D").all() for h, f, _ in blocks)
    frames = tags[(tags["code"] == "SET PIECES") & (tags["Type"].astype(str).str.lower() == "kick off (start)")]["start_frame"]
    n = {h: int(((frames >= f) & (frames <= z)).sum()) for h, f, z in blocks}
    ok = ok and n[1] <= 1 and n[2] == 1 and t0[2] > blocks[1][1]
    verdict("A3", bool(ok))
    # A4, A5
    verdict("A4", a4(lab, tags, data.team_tags(), blocks))
    verdict("A5", a5(lab, play, P, slot_team, B, pitch))
    # A6
    model = infer.load_model(weights)
    if with_a6:
        from analysis.clean import equiv
        sw, mi, rot = equiv.agreement(P, B, slot_team, block, t0, pitch, model)
        verdict("A6r", rot == 1.0)
        verdict("A6s", sw >= reg["A6"]["swap"])
        verdict("A6m", mi >= reg["A6"]["mirror"])
    # A7
    team = np.where(np.isfinite(P[..., 0]), slot_team[None, :], -1)
    right = features.home_attacks_right(P, team, play)
    xml = data.home_attacks_right()
    verdict("A7", right[1] == xml[1] and right[2] == xml[2] and right[1] != right[2])
    # A8, A9
    st = out_stats(lab, blocks, t0)
    verdict("A8", in_band(st, reg["B1"]))
    verdict("A9", in_band(st, reg["B2"]))
    # A10, A11
    verdict("A10", a10(P, slot_team, B, block, play, pitch, reg, model, reg["A10"]))
    verdict("A11", a11(P, slot_team, B, play, pitch, reg, model[1]))
    # A14
    verdict("A14", len(tags) > 0 and not tags["code"].isna().any())
    return 0


if __name__ == "__main__":
    sys.exit(main())
