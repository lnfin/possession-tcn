"""Step B0: reading and units of the match's inputs. Prints only I1 PASS ... I4 FAIL.

    python -m analysis.clean.check_inputs QUANTILES.json [--blocks JSON]

No network, no CSV, no number on the screen or on disk. --blocks: the fixture only.
"""

import json
import sys

import numpy as np

from mtah import data
from mtah.transfer import features, ours


def side_ok(players, ball, slot_team, block_list, t0):
    """I1 / A7: home's side per half, read from the blocks' mean x, as the XML says; halves differ."""
    period = ours.play_period(ours.block_of(len(ball), block_list), t0, block_list)
    team = np.where(np.isfinite(players[..., 0]), slot_team[None, :], -1)
    right = features.home_attacks_right(players, team, period)
    xml = data.home_attacks_right()
    return right[1] == xml[1] and right[2] == xml[2] and right[1] != right[2]


def tags_ok(tags, block_list):
    """I2 / A14: all codes read, Kick Off (Start) rows parse, frames fall into the file."""
    if not len(tags) or tags["code"].isna().any() or "Type" not in tags:
        return False
    ours.play_starts(tags, block_list)
    return bool(tags["start_frame"].between(0, 10 ** 7).all())


def units_ok(raw_players, raw_ball, block):
    """I3: coordinates inside [-0.2, 1.2] on >= 99 % of the observations, speed column < 12 m/s."""
    inside = block > 0
    B = raw_ball[inside]
    P = raw_players[inside]
    def share(x):
        x = x[np.isfinite(x)]
        return float(((x >= -0.2) & (x <= 1.2)).mean())
    s = P[..., 2][np.isfinite(P[..., 2])]
    return share(B) >= 0.99 and share(P[..., :2]) >= 0.99 and float((s < 12).mean()) >= 0.99


def coverage_ok(X, period, q):
    """I4: >= 95 % of the frames inside p0.1 - p99.9 of the M4 file, column by column."""
    X = X[period > 0]
    lo, hi = np.array(q["lo"]), np.array(q["hi"])
    ok = True
    for k in range(X.shape[1]):
        x = X[:, k]
        x = x[np.isfinite(x)]
        if len(x) and float(((x >= lo[k]) & (x <= hi[k])).mean()) < 0.95:
            ok = False
    return ok


def main():
    q = json.loads(open(sys.argv[1]).read())
    blocks = ours.blocks()
    if "--blocks" in sys.argv:
        blocks = [tuple(b) for b in json.loads(sys.argv[sys.argv.index("--blocks") + 1])]
    raw_players, raw_ball, slot_team = data.load_tracking()
    pitch = data.pitch_m()
    scale = np.array(pitch, dtype=np.float64)
    P = np.asarray(raw_players[..., :2], dtype=np.float64) * scale
    B = np.asarray(raw_ball, dtype=np.float64) * scale
    tags = data.load_smart_tagging()
    i2 = tags_ok(tags, blocks)
    t0 = ours.play_starts(tags, blocks) if i2 else {h: f for h, f, _ in blocks}
    block = ours.block_of(len(B), blocks)
    i1 = side_ok(P, B, slot_team, blocks, t0)
    i3 = units_ok(raw_players, raw_ball, block)
    period = ours.play_period(block, t0, blocks)
    team = np.where(np.isfinite(P[..., 0]), slot_team[None, :], -1).astype(np.int8)
    X, _, _ = features.build(P, team, B, period, pitch, features.ALL_RADII)
    i4 = coverage_ok(X, period, q)
    for name, ok in (("I1", i1), ("I2", i2), ("I3", i3), ("I4", i4)):
        print(name, "PASS" if ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
