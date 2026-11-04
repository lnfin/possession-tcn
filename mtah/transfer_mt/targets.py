"""Boundary targets of C1, per period, from the labels of a copy only.

  sw   a direct H <-> A change within +-K_BOUND frames (the possession boundary)
  lv   a live <-> dead change within +-K_BOUND frames
"""

import numpy as np

AUX = ("sw", "lv")
K_BOUND = 12             # TAUS[0] of the official scorer


def changes(y):
    """[T] bool: y[t] != y[t-1], both labelled (y >= 0)."""
    y = np.asarray(y)
    out = np.zeros(len(y), dtype=bool)
    out[1:] = (y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0)
    return out


def distance_to(flag, cap=10 ** 6):
    """[T] frames to the nearest True of flag (cap where there is none)."""
    idx = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(flag, idx, 10 ** 9)[::-1])[::-1]
    return np.minimum(np.minimum(idx - last, nxt - idx), cap)


def _segment_targets(y):
    """Boundary targets of one period: dict name -> (target, weight)."""
    lab = y >= 0
    ch = changes(y)
    prev = np.r_[-1, y[:-1]]
    sw_ev = ch & np.isin(y, (0, 1)) & np.isin(prev, (0, 1))
    lv_ev = ch & ~sw_ev
    return {"sw": ((distance_to(sw_ev) <= K_BOUND) * 1.0, lab * 1.0),
            "lv": ((distance_to(lv_ev) <= K_BOUND) * 1.0, lab * 1.0)}
