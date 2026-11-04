"""The decoder's plateau rule, section 4.5, on a score table of grid.py.

Everything here is a pure function of the table, the ε and the units in play. W over a set of units is
1/2 the mean of Metrica plus 1/2 the mean of IDSSE, each with the weights w_u; on all units that is W.
"""

import itertools
import math

import numpy as np
from scipy import ndimage

from analysis.clean import grid, metric

CS = (0.5, 1.0, 2.0)


def weights(units, source=None):
    """Weights over UNIT_NAMES of the summary on `units`; source M / I: that source alone."""
    w = np.zeros(len(metric.UNIT_NAMES))
    for src, keep in metric.SOURCE_UNITS.items():
        if source not in (None, src):
            continue
        idx = [metric.UNIT_NAMES.index(u) for u in keep if u in units]
        den = sum(metric.UNIT_W[metric.UNIT_NAMES[i]] for i in idx)
        share = 1.0 if source else 0.5
        for i in idx:
            w[i] = share * metric.UNIT_W[metric.UNIT_NAMES[i]] / den
    return w


def n_eff(units, source):
    w = weights(units, source)
    return 1.0 / float((w ** 2).sum())


def eps_source(eps_s, units, source):
    return eps_s * math.sqrt(4.8 / n_eff(units, source))


class Lattice:
    """The grid as an n-dimensional array over its axes; kappa = 0 present on every switch_min level."""

    def __init__(self, axes, table):
        self.axes = [tuple(a) for a in axes]
        self.shape = tuple(len(a) for a in axes)
        self.table = table
        self.c1 = len(axes) == 4
        have = table.index()
        self.at = np.full(self.shape, -1, dtype=np.int64)
        for idx in np.ndindex(self.shape):
            val = self.value(idx)
            if val[0] <= 2 * val[1]:
                self.at[idx] = have[grid.canon(*val)]
        self.adm = self.at >= 0

    def value(self, idx):
        v = [self.axes[k][i] for k, i in enumerate(idx)]
        return (v[0], v[1], v[2], v[3]) if self.c1 else (v[0], v[1], 0.0, 0.0)

    def nd(self, x):
        """x [cells, ...] -> [lattice..., ...], NaN on inadmissible cells."""
        out = np.full(self.shape + x.shape[1:], np.nan)
        out[self.adm] = x[self.at[self.adm]]
        return out

    def _level(self, axis, value):
        """Index of the nearest level in the log, ties to the smaller; None if outside the axis."""
        lev = self.axes[axis]
        if value < lev[0] or value > lev[-1]:
            return None
        d = [abs(math.log(value / v)) for v in lev]
        best = min(d)
        return next(i for i, x in enumerate(d) if x <= best + 1e-12)

    def shifted(self, idx, c):
        """The cell (switch c, dead c, kappa, switch_min c); None outside the grid."""
        out = list(idx)
        for axis in ((0, 1, 3) if self.c1 else (0, 1)):
            j = self._level(axis, self.axes[axis][idx[axis]] * c)
            if j is None:
                return None
            out[axis] = j
        out = tuple(out)
        return out if self.adm[out] else None


def plateau(S, adm, eps):
    """Connected (Chebyshev) set of cells within eps of the maximum, containing the maximum."""
    best = np.nanmax(S)
    with np.errstate(invalid="ignore"):
        near = adm & (S >= best - eps)
    lab, _ = ndimage.label(near, structure=np.ones((3,) * S.ndim))
    keep = set(lab[adm & (S == best)].tolist())
    return np.isin(lab, sorted(keep))


def _scores(lat, table, units, source=None, t2_only=False):
    """Per-cell summaries on the lattice from the table: S, S_e [..., e], and the source / T2 versions."""
    w = weights(units, source)
    X = table.S[:, :, 2:] if t2_only else table.S                       # [cells, 9, copies, e]
    unit_e = grid.unit_scores(np.moveaxis(X.mean(2), 1, 0))             # [6, cells, e]
    Se = lat.nd((w[:, None, None] * unit_e).sum(0))
    return Se.mean(-1), Se


def select(lat, units, eps_s, source=None):
    """Steps 1-4 on `units`. source: step P8, everything on the training source alone."""
    S, Se = _scores(lat, lat.table, units, source)
    P = plateau(S, lat.adm, eps_s)
    notes = []
    if source is None:
        PM = plateau(_scores(lat, lat.table, units, "M")[0], lat.adm, eps_source(eps_s, units, "M"))
        PI = plateau(_scores(lat, lat.table, units, "I")[0], lat.adm, eps_source(eps_s, units, "I"))
        PT = plateau(_scores(lat, lat.table, units, t2_only=True)[0], lat.adm, eps_s)
        A = P & PM & PI & PT
        if not A.any():
            A = P
            notes.append("semantics diverge")
    else:
        A = plateau(S, lat.adm, eps_source(eps_s, units, source))
    best_e = np.nanmax(Se.reshape(-1, Se.shape[-1]), axis=0)

    def r_c(idx, cs):
        vals = []
        for c in cs:
            j = lat.shifted(idx, c)
            if j is None:
                return None
            vals.append(float((Se[j] - best_e).min()))
        return min(vals)

    cells = [tuple(int(v) for v in i) for i in np.argwhere(A)]
    rc = {i: r_c(i, CS) for i in cells}
    part = {i: v for i, v in rc.items() if v is not None}
    if not part:
        notes.append("no shift inside the grid: r_c at c = 1 only")
        part = {i: r_c(i, (1.0,)) for i in cells}
    star = max(part.values())
    Q = {i for i, v in part.items() if v >= star - eps_s}
    pick = _deepest(lat, Q)
    inner = sorted(i for i in cells if all(lat.shifted(i, c) is not None and A[lat.shifted(i, c)] for c in CS))
    return {"idx": pick, "value": lat.value(pick), "set": A, "r_star": star, "notes": notes,
            "P_scale": [lat.value(i) for i in inner], "diag_width": _diagonal(lat, pick, A),
            "n_set": len(cells), "n_Q": len(Q), "S_star": float(np.nanmax(S))}


def _diagonal(lat, idx, A):
    """Steps along (switch, dead) x c from the cell, both ways, while the shifted cell stays in the set."""
    width = 1
    for c in (0.5, 2.0):
        i = idx
        while True:
            j = lat.shifted(i, c)
            if j is None or j == i or not A[j]:
                break
            width += 1
            i = j
    return width


def _deepest(lat, Q):
    """The cell of Q furthest (Chebyshev) from anything outside Q; ties by the rule of step 4."""
    n = len(lat.shape)
    kappa_axis = 2 if lat.c1 else None
    ranges = [range(0 if k == kappa_axis else -1, s + 1) for k, s in enumerate(lat.shape)]
    outside = np.array([i for i in itertools.product(*ranges)
                        if i not in Q], dtype=np.int64).reshape(-1, n)
    best, key = None, None
    for i in sorted(Q):
        dist = int(np.abs(outside - np.array(i)).max(1).min()) if len(outside) else 0
        v = lat.value(i)
        k = (dist, v[0], v[1], -v[2], v[3])
        if key is None or k > key:
            best, key = i, k
    return best


def extended_axes(lat, pick):
    """Axes after the one-pass extension of step 5 for the cell chosen on all matches."""
    axes = [list(a) for a in lat.axes]
    changed = []
    val = lat.value(pick)
    for axis, lev in enumerate(lat.axes):
        i = pick[axis]
        scaled = axis != 2
        if axis == 3 and val[2] == 0:
            continue                                         # kappa = 0: no edge on this axis
        low = i == 0 or (scaled and lev[i - 1] * 0.5 < lev[0])
        high = i == len(lev) - 1 or (scaled and lev[i + 1] * 2.0 > lev[-1])
        if axis == 2:
            low = False                                      # kappa = 0 is a physical bound
        if high:
            axes[axis].append(lev[-1] * lev[-1] / lev[-2])
            changed.append((axis, "high", axes[axis][-1]))
        if low:
            axes[axis].insert(0, lev[0] * lev[0] / lev[1])
            changed.append((axis, "low", axes[axis][0]))
    return axes, changed


def nested(lat, eps_s):
    """{unit: cell chosen without it} on the final grid."""
    return {u: select(lat, tuple(x for x in metric.UNIT_NAMES if x != u), eps_s)
            for u in metric.UNIT_NAMES}
