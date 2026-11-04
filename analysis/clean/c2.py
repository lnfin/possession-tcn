"""Values of class Р of candidate C2, table 1.3: distributions on copies G, no metric involved.

A transition k is the first frame of H or A after D by the copy's labels. It counts only if the ball
is seen on every frame of [k - 25, k + 12). Pooled from the six IDSSE matches, without J03WR9, without
the held-out unit, unweighted. In the direction Metrica -> IDSSE of step P8 the same on Metrica's copies.
"""

import numpy as np

from analysis.clean import metric

BEFORE, AFTER = 25, 12
Q_V_STILL, Q_T_STILL, Q_V_KICK, Q_D_PLAYER = 90, 10, 10, 90


def pool_matches(hold):
    """The matches of the pool of a fold."""
    if hold == "src:M":
        return metric.METRICA
    six = metric.GAMES_EVAL[3:]
    if hold in ("all", "src:I"):
        return six
    drop = metric.UNIT_HOLD[hold]
    return tuple(n for n in six if n not in drop)


def transitions(y, period, p):
    """Frames k as above; y the copy's labels R."""
    ks = []
    for per in (1, 2):
        rows = np.flatnonzero(period == per)
        a, b = rows[0], rows[-1] + 1
        yy = y[a:b]
        for k in np.flatnonzero((yy[:-1] == 2) & np.isin(yy[1:], (0, 1))) + 1 + a:
            if k - BEFORE >= a and k + AFTER <= b and p[k - BEFORE:k + AFTER].all():
                ks.append(int(k))
    return ks


def compute(rows):
    """{v_still, T_still, v_kick, d_player} from [(v, dmin, ks)]."""
    pool = []
    for v, _, ks in rows:
        pool += [v[k - BEFORE:k] for k in ks]
    pool = np.concatenate(pool).astype(np.float64)
    v_still = float(np.percentile(pool[np.isfinite(pool)], Q_V_STILL, method="linear"))
    lengths, kicks, dist = [], [], []
    for v, dmin, ks in rows:
        for k in ks:
            n = 0
            while k - 1 - n >= 0 and v[k - 1 - n] < v_still:
                n += 1
            lengths.append(n)
            kicks.append(float(np.nanmax(v[k:k + AFTER])))
            if np.isfinite(dmin[k]):
                dist.append(float(dmin[k]))
    T_still = max(1, int(np.ceil(np.percentile(lengths, Q_T_STILL, method="linear"))))
    return {"v_still": v_still, "T_still": T_still,
            "v_kick": float(np.percentile(kicks, Q_V_KICK, method="linear")),
            "d_player": float(np.percentile(dist, Q_D_PLAYER, method="linear")),
            "n_transitions": len(kicks)}


def values(root, hold, seeds, only=()):
    """The values of a fold; only: the smoke of step M4, these matches."""
    rows = []
    for name in pool_matches(hold):
        if only and name not in only:
            continue
        for s in seeds:
            base = "{}/{}_g1_s{}".format(root, name, s)
            f = np.load(base + ".f.npy", mmap_mode="r")
            i = np.load(base + ".i.npy", mmap_mode="r")
            y = np.load(base + ".y.npy", mmap_mode="r")[:, metric.REFS.index("R")]
            period = np.load(base + ".per.npy")
            v, dmin, p = np.array(f[:, 0]), np.array(f[:, 1]), np.array(i[:, 1])
            rows.append((v, dmin, transitions(np.array(y), period, p)))
    return compute(rows)
