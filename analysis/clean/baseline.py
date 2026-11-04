"""The official nearest-player baseline, vectorised, and the fallback base* of section 9.2.

smooth is a copy of starter/baseline.py; codes follows nearest_player_labels on arrays, so that no
ATD-format file is needed on the copies. tests/test_clean.py compares them on a small synthetic file.

    python -m analysis.clean.baseline           # base* on the 9 clean open matches, by W

The fallback reads only the external matches' files (journal, before the run).
"""

import json
import sys

import numpy as np

from analysis.clean import metric
from mtah.external import reference

HOME, AWAY, NOBODY = 0, 1, -1
WINDOWS = (25, 50, 125)


def smooth(codes, window):
    """Majority vote over a centred window of +/- `window` frames. D where nobody has the ball.
    Ties go to the home side. Copy of the official function."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def codes_of(players_m, team, ball_m):
    """int8 [T]: team of the first closest tracked player, NOBODY without ball or players.
    team [S] by slot or [T, S] by frame (-1: not tracked)."""
    P = np.asarray(players_m, dtype=np.float64)
    B = np.asarray(ball_m, dtype=np.float64)
    team = np.asarray(team)
    if team.ndim == 1:
        team = np.broadcast_to(team[None, :], P.shape[:2])
    d = ((P - B[:, None, :]) ** 2).sum(-1)
    d = np.where(np.isfinite(d) & (team >= 0), d, np.inf)
    j = d.argmin(1)
    r = np.arange(len(B))
    return np.where(np.isfinite(d[r, j]), team[r, j], NOBODY).astype(np.int8)


def labels_of(codes, periods, window):
    """Chars [T]: smooth applied to each period on its own, X elsewhere."""
    out = np.full(len(codes), "X", dtype="<U1")
    for _, a, b in periods:
        out[a:b + 1] = smooth(codes[a:b + 1], window)
    return out


def base_star_clean(ref_params):
    """{window: W(mean spell over E)} on the clean matches; the fallback of section 9.2."""
    res = {}
    for w in WINDOWS:
        unit_scores = {}
        for u, names in metric.UNITS.items():
            per_match = []
            for name in names:
                G = metric.load_game(name)
                Y, _, _ = metric.references(name, G, ref_params)
                L, H = G["pitch_m"]
                c = codes_of(np.asarray(G["players"], dtype=np.float64) * np.array([L, H]), G["slot_team"],
                             np.asarray(G["ball"], dtype=np.float64) * np.array([L, H]))
                p = labels_of(c, G["periods"], w)
                per_match.append(np.mean([metric.spell_macro(reference.to_chars(Y[e]), p, G["periods"])[0]
                                          for e in metric.REFS]))
            unit_scores[u] = float(np.mean(per_match))
        res[w] = metric.W(unit_scores)
    return res


def main():
    res = base_star_clean(reference.RefParams(out_lag=int(sys.argv[1]), cluster=int(sys.argv[2])))
    print(json.dumps({"scores": res, "base_star": max(res, key=res.get)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
