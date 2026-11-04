"""Viterbi over H, A, D per period, with a team-switch cost that may follow the boundary head.

The cost of an H <-> A change entering frame t is

    switch_t = max(switch_min, switch - kappa * logit(b_t))

with b_t the boundary head's P(an H <-> A change within +-K_BOUND frames). kappa = 0 is a constant
cost. H <-> team -> D and back cost `dead`. All values are chosen on the open matches' out-of-fold
predictions.
"""

import numpy as np

from ..seqmodel import decode as seqdecode

LABEL_CHARS = np.array(["H", "A", "D"])


def switch_costs(switch, kappa, boundary, switch_min):
    """[T] cost of an H <-> A change entering each frame (a float when kappa is 0)."""
    if not kappa or boundary is None:
        return float(switch)
    F = seqdecode.FLOOR
    b = np.clip(np.round(np.asarray(boundary, dtype=np.float64), seqdecode.ROUND), F, 1 - F)
    return np.maximum(switch_min, switch - kappa * np.log(b / (1 - b)))


def viterbi3(cost, sw, to_dead, from_dead):
    """3-state (H, A, D) min-cost path; sw a float or [T] switch cost entering each frame."""
    T = len(cost)
    c0, c1, c2 = (cost[:, k].tolist() for k in range(3))
    s = [float(sw)] * T if np.isscalar(sw) else np.asarray(sw, dtype=np.float64).tolist()
    a0, a1, a2 = c0[0], c1[0], c2[0]
    back = bytearray(3 * T)
    td, fd = float(to_dead), float(from_dead)
    for t in range(1, T):
        st = s[t]
        # into H: from H, A (+switch), D (+from_dead); first minimum wins as in np.argmin
        x0, x1, x2 = a0, a1 + st, a2 + fd
        if x0 <= x1 and x0 <= x2:
            n0, b0 = x0, 0
        elif x1 <= x2:
            n0, b0 = x1, 1
        else:
            n0, b0 = x2, 2
        x0, x1, x2 = a0 + st, a1, a2 + fd
        if x0 <= x1 and x0 <= x2:
            n1, b1 = x0, 0
        elif x1 <= x2:
            n1, b1 = x1, 1
        else:
            n1, b1 = x2, 2
        x0, x1, x2 = a0 + td, a1 + td, a2
        if x0 <= x1 and x0 <= x2:
            n2, b2 = x0, 0
        elif x1 <= x2:
            n2, b2 = x1, 1
        else:
            n2, b2 = x2, 2
        a0, a1, a2 = n0 + c0[t], n1 + c1[t], n2 + c2[t]
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = b0, b1, b2
    acc = [a0, a1, a2]
    path = np.empty(T, dtype=np.int8)
    k = int(np.argmin(acc))
    path[-1] = k
    for t in range(T - 1, 0, -1):
        k = back[3 * t + k]
        path[t - 1] = k
    return path


def decode_periods(P, period, switch, dead, kappa, boundary, switch_min):
    """Labels H/A/D per period (X outside) of P [T, 3]; boundary [T] = P(switch) or None."""
    P = seqdecode.rounded(np.where(np.isfinite(P), P, 0.0))
    out = np.full(len(P), "X", dtype="<U1")
    for per in (1, 2):
        rows = np.flatnonzero(period == per)
        if not len(rows):
            continue
        a, b = rows[0], rows[-1] + 1
        sw = switch_costs(switch, kappa, None if boundary is None else boundary[a:b], switch_min)
        out[a:b] = LABEL_CHARS[viterbi3(-np.log(P[a:b]), sw, dead, dead)]
    return out
