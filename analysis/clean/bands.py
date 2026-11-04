"""out_lag, cluster, tables 0.4 and 6.2, band B1: only events and labels of the open matches.

    python -m analysis.clean.bands              # prints the values and the tables

No file of the match, no model. On step M1a the printed out_lag and cluster go to the journal; on P2
out_lag_value and cluster_value run again and values.json asserts they are the same.
"""

import json
import math
import sys

import numpy as np

from analysis.clean import metric
from mtah.external import metrica_open, reference

FPS = 25
LS = (0, 12, 25, 37, 50, 75, 100, 125, 150)


def out_lag_value(games):
    """Median over all BALL OUT events: end frame - first frame past the last in-bounds position of
    the clean ball, looked for between the event's start and end + 25."""
    lags = []
    for G in games:
        b = G["ball"]
        for r in G["events"][G["events"].type == "BALL OUT"].itertuples():
            seg = b[int(r.start):int(r.end) + FPS + 1]
            ok = np.isfinite(seg[:, 0]) & (seg[:, 0] >= 0) & (seg[:, 0] <= 1) & (seg[:, 1] >= 0) & (seg[:, 1] <= 1)
            idx = np.flatnonzero(ok)
            if len(idx):
                lags.append(int(r.end) - (int(r.start) + int(idx[-1]) + 1))
    return int(np.median(lags)), len(lags)


def cluster_value(games):
    """p95, rounded up, of the gap between a BALL OUT and a BALL LOST adjacent in the event stream
    without CARD and SUBSTITUTION."""
    gaps = []
    for G in games:
        E = G["events"].sort_values(["start", "end"], kind="stable")
        E = E[~E.type.isin(["CARD", "SUBSTITUTION"])]
        st, ty, pr = E.start.to_numpy(), E.type.to_numpy(), E.period.to_numpy()
        gaps += [int(st[i + 1] - st[i]) for i in range(len(st) - 1)
                 if {ty[i], ty[i + 1]} == {"BALL OUT", "BALL LOST"} and pr[i] == pr[i + 1]]
    return int(math.ceil(np.percentile(gaps, 95))), len(gaps)


def stats(y, per):
    """(possessions per minute, median length s, share < 1 s, share D) from the start of the periods."""
    lens, frames, dead = [], 0, 0
    for _, a, b in per:
        s = y[a:b + 1]
        frames += len(s)
        dead += int((s == 2).sum())
        cut = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1, len(s)]
        lens += [(v - u) / FPS for u, v in zip(cut[:-1], cut[1:]) if s[u] != 2]
    lens = np.array(lens)
    return len(lens) / (frames / FPS / 60), float(np.median(lens)), float((lens < 1).mean()), dead / frames


def band_b1(rows):
    """[min - 25 %, max + 25 %] of the four quantities; rows: (poss/min, median, < 1 s, D) by E and match."""
    r = np.array(rows)
    return {"poss_per_min": [float(r[:, 0].min() * 0.75), float(r[:, 0].max() * 1.25)],
            "median_s": [float(r[:, 1].min() * 0.75), float(r[:, 1].max() * 1.25)],
            "short_share": [0.0, float(r[:, 2].max() * 1.25)],
            "dead_share": [float(r[:, 3].min() * 0.75), float(r[:, 3].max() * 1.25)]}


def reference_set(ref_params):
    """{match: ({R, R25, R75, Rdef}, periods, info)} of the 10 matches."""
    out = {}
    for name in metric.GAMES_TRAIN:
        G = metric.load_game(name)
        Y, _, info = metric.references(name, G, ref_params)
        out[name] = (Y, G["periods"], info)
    return out


def _spell(g, p, per):
    return metric.spell_macro(reference.to_chars(g), reference.to_chars(p), per)[0]


def main():
    games = [metrica_open.load(g) for g in metric.METRICA]
    lag, n_lag = out_lag_value(games)
    clu, n_clu = cluster_value(games)
    print("out_lag", lag, "n", n_lag, " cluster", clu, "pairs", n_clu)
    rp = reference.RefParams(out_lag=lag, cluster=clu)
    Y = reference_set(rp)

    print("\nTable 6.2: match | R: poss/min, median s, < 1 s, D | R(25) | R(75) | R(100): poss/min, median s")
    rows = []
    for name, (y, per, _) in Y.items():
        c = [stats(y[k], per) for k in ("R", "R25", "R75")]
        c.append(stats(reference.merge(y["R"], per, 100), per))
        d = stats(y["Rdef"], per)
        rows += c[:3] + [d]
        print("{:9s} {:.2f} {:.2f} {:.3f} {:.3f} | {:.2f} {:.2f} | {:.2f} {:.2f} | {:.2f} {:.2f} | Rdef {:.2f} {:.2f}".format(
            name, *c[0], *c[1][:2], *c[2][:2], *c[3][:2], *d[:2]))
    print("\nB1 (E_def):", json.dumps(band_b1(rows)))
    print("IDSSE lag*:", {k: (v[2]["lag"], round(v[2]["share"], 3)) for k, v in Y.items() if v[2]})

    print("\nTarget of a perfect R(b), E_def: b =", LS)
    S = {}
    for name in metric.GAMES_EVAL:
        y, per, _ = Y[name]
        Rb = {b: reference.merge(y["R"], per, b) for b in LS}
        S[name] = {(e, b): _spell(y[e], Rb[b], per) for e in metric.REFS for b in LS}

    def w(f):
        unit = {u: np.mean([f(m) for m in ms]) for u, ms in metric.UNITS.items()}
        return metric.W(unit)
    se = {(e, b): w(lambda m: S[m][(e, b)]) for e in metric.REFS for b in LS}
    print("mean ", " ".join("{:.3f}".format(np.mean([se[(e, b)] for e in metric.REFS])) for b in LS))
    print("worst", " ".join("{:.3f}".format(min(se[(e, b)] for e in metric.REFS)) for b in LS))
    best = {e: max(se[(e, b)] for b in LS) for e in metric.REFS}
    print("r    ", " ".join("{:+.3f}".format(min(se[(e, b)] - best[e] for e in metric.REFS)) for b in LS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
