"""Rounding of the probabilities and the min-cost path.

Probabilities are rounded to ROUND decimals first, so tiny numeric differences cannot flip a path.
"""

import numpy as np

ROUND = 4
FLOOR = 10.0 ** -ROUND


def rounded(P):
    return np.maximum(np.round(np.asarray(P, dtype=np.float64), ROUND), FLOOR)


def viterbi(cost, trans):
    """Min-cost state path for cost [T, S] and transition costs trans [S, S] (from, to)."""
    T, S = cost.shape
    acc = cost[0].copy()
    back = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        cand = acc[:, None] + trans                  # [from, to]
        back[t] = np.argmin(cand, axis=0)
        acc = cand[back[t], np.arange(S)] + cost[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(np.argmin(acc))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
