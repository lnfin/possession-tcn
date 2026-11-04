"""Decoder cells scored on out-of-fold predictions: the raw material of sections 4.5 and 4.6.

A cell is (switch, dead, kappa, switch_min); kappa = 0 is one cell, whatever switch_min. A score table
holds, per cell, spell and macro over [match of GAMES_EVAL, copy (T1 s0, T1 s1, T2 s0, T2 s1), reference].
"""

import multiprocessing as mp
import sys

import numpy as np

from analysis.clean import baseline, metric
from mtah.external import reference
from mtah.transfer_mt import decode

SWITCH = (8.0, 12.0, 16.0, 24.0, 36.0, 48.0, 64.0, 96.0)
DEAD = (12.0, 16.0, 24.0, 36.0, 48.0, 64.0, 96.0)
KAPPA = (0.0, 2.0, 4.0, 8.0)
SWITCH_MIN = (0.5, 1.0, 2.0, 4.0, 8.0)
COPIES = ((2, 0), (2, 1), (3, 0), (3, 1))


def canon(switch, dead, kappa, switch_min):
    """The key of a cell: switch_min does not exist at kappa = 0."""
    return (float(switch), float(dead), float(kappa), float(switch_min) if kappa else 0.0)


def base_axes(c1):
    return [SWITCH, DEAD, KAPPA, SWITCH_MIN] if c1 else [SWITCH, DEAD]


def all_cells(axes):
    out = set()
    sw, de = axes[0], axes[1]
    kap = axes[2] if len(axes) > 2 else (0.0,)
    smin = axes[3] if len(axes) > 2 else (0.0,)
    for s in sw:
        for d in de:
            if s > 2 * d:
                continue
            for k in kap:
                for m in smin:
                    out.add(canon(s, d, k, m))
    return sorted(out)


class Oof:
    """Mean probability of the given seeds per (match, copy), lazily from oof_<variant>_<unit>_s<seed>.npz."""

    def __init__(self, folder, variant, seeds, tag=""):
        self.folder, self.variant, self.seeds, self.tag = folder, variant, seeds, tag

    def unit_of(self, name):
        return next(u for u, ms in metric.UNITS.items() if name in ms)

    def get(self, name, g, s):
        key = "{}_g{}_s{}".format(name, g, s)
        Ps, ls = [], []
        for seed in self.seeds:
            path = "{}/oof_{}{}_{}_s{}.npz".format(self.folder, self.variant, self.tag, self.unit_of(name), seed)
            with np.load(path) as z:
                Ps.append(z[key + "/P"].astype(np.float64))
                if key + "/sw" in z.files:
                    ls.append(z[key + "/sw"].astype(np.float64))
        P = np.mean(Ps, axis=0)
        b = 1.0 / (1.0 + np.exp(-np.mean(ls, axis=0))) if ls else None
        return P, b


def _gt(root, name, g, s):
    y = np.load("{}/{}_g{}_s{}.y.npy".format(root, name, g, s))
    period = np.load("{}/{}_g{}_s{}.per.npy".format(root, name, g, s))
    return y, period


def periods_of(period):
    out = []
    for per in (1, 2):
        rows = np.flatnonzero(period == per)
        out.append((per, int(rows[0]), int(rows[-1])))
    return out


def _score_job(args):
    root, oof, name, g, s, cells = args
    y, period = _gt(root, name, g, s)
    per = periods_of(period)
    pm = metric.load_pm()
    runs = metric.runs_of(per)
    gcodes = [pm.label_codes(reference.to_chars(y[:, e])) for e in range(y.shape[1])]
    P, b = oof.get(name, g, s)
    S = np.zeros((len(cells), y.shape[1]))
    F = np.zeros_like(S)
    for i, (sw, de, ka, sm) in enumerate(cells):
        lab = decode.decode_periods(P, period, sw, de, ka, b, sm)
        pc = pm.label_codes(lab)
        filled = pm.fill_abstentions(pc, runs)
        for e, gc in enumerate(gcodes):
            S[i, e] = pm.spell_scores(gc, filled, runs)["mean_f1"]
            F[i, e] = pm.macro_f1(pm.scored_confusion(pm.confusion(gc, pc)))
    return name, g, s, S, F


def pool(workers):
    return mp.get_context("fork" if sys.platform.startswith("linux") else "spawn").Pool(workers)


def score(root, oof, cells, workers):
    """(S, F): float64 [len(cells), 9, 4, 4]."""
    jobs = [(root, oof, name, g, s, cells) for name in metric.GAMES_EVAL for g, s in COPIES]
    S = np.zeros((len(cells), len(metric.GAMES_EVAL), len(COPIES), len(metric.REFS)))
    F = S.copy()
    with pool(workers) as p:
        for name, g, s, a, b in p.imap_unordered(_score_job, jobs):
            i, j = metric.GAMES_EVAL.index(name), COPIES.index((g, s))
            S[:, i, j], F[:, i, j] = a, b
    return S, F


def _base_job(args):
    root, name, g, s, w = args
    y, period = _gt(root, name, g, s)
    per = periods_of(period)
    codes = np.load("{}/{}_g{}_s{}.b.npy".format(root, name, g, s))
    lab = baseline.labels_of(codes, per, w)
    out = [metric.spell_macro(reference.to_chars(y[:, e]), lab, per) for e in range(y.shape[1])]
    return name, g, s, w, np.array(out)


def score_baseline(root, workers):
    """{window: (S [9, 4, 4], F [9, 4, 4])} of the three official variants on the same copies."""
    jobs = [(root, n, g, s, w) for n in metric.GAMES_EVAL for g, s in COPIES for w in baseline.WINDOWS]
    out = {w: (np.zeros((9, 4, 4)), np.zeros((9, 4, 4))) for w in baseline.WINDOWS}
    with pool(workers) as p:
        for name, g, s, w, r in p.imap_unordered(_base_job, jobs):
            i, j = metric.GAMES_EVAL.index(name), COPIES.index((g, s))
            out[w][0][i, j] = r[:, 0]
            out[w][1][i, j] = r[:, 1]
    return out


class Table:
    """Cells -> scores; grows when an axis is extended."""

    def __init__(self):
        self.cells, self.S, self.F = [], None, None

    def add(self, cells, S, F):
        self.cells += list(cells)
        self.S = S if self.S is None else np.concatenate([self.S, S])
        self.F = F if self.F is None else np.concatenate([self.F, F])

    def index(self):
        return {c: i for i, c in enumerate(self.cells)}

    def missing(self, cells):
        have = self.index()
        return [c for c in cells if c not in have]

    def save(self, path):
        np.savez(path, cells=np.array(self.cells), S=self.S, F=self.F)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        t = cls()
        t.add([tuple(float(v) for v in c) for c in z["cells"]], z["S"], z["F"])
        return t


def unit_scores(x):
    """x [..., 9, ...]: mean per unit along the match axis given as axis 0 -> [6, ...]."""
    out = []
    for u in metric.UNIT_NAMES:
        idx = [metric.GAMES_EVAL.index(n) for n in metric.UNITS[u]]
        out.append(x[idx].mean(0))
    return np.stack(out)
