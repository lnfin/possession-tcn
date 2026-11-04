"""Gates of section 4.6 and the noise of 4.2, on nested unit scores.

A model's evidence is `Nested`: for every unit its s_u, f_u, s_u^e and its T2-only s_u from the decoder
chosen without that unit.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from analysis.clean import grid, metric, plateau

THRESH = 1.0 / 64.0
UNITS = metric.UNIT_NAMES


@dataclass
class Nested:
    s: np.ndarray           # [6]
    f: np.ndarray           # [6]
    se: np.ndarray          # [6, 4]
    t2: np.ndarray          # [6]


def nested_scores(table, picks):
    """Nested of a model; picks {unit: cell key}, its own decoder for each unit."""
    have = table.index()
    s, f, se, t2 = [], [], [], []
    for k, u in enumerate(UNITS):
        i = have[picks[u]]
        S = grid.unit_scores(table.S[i])[k]                 # [copies, e]
        F = grid.unit_scores(table.F[i])[k]
        s.append(S.mean())
        f.append(F.mean())
        se.append(S.mean(0))
        t2.append(S[2:].mean())
    return Nested(np.array(s), np.array(f), np.array(se), np.array(t2))


def unit_at(table, key):
    """Nested with one decoder for all units."""
    return nested_scores(table, {u: key for u in UNITS})


def baseline_nested(S, F):
    """Nested of the baseline: S, F [9, 4, 4]."""
    u = grid.unit_scores(S)
    uf = grid.unit_scores(F)
    return Nested(u.mean((1, 2)), uf.mean((1, 2)), u.mean(1), u[:, 2:].mean((1, 2)))


def W(x, units=UNITS):
    return float((plateau.weights(units) * x).sum())


def W_src(x, source, units=UNITS):
    return float((plateau.weights(units, source) * x).sum())


def signflip_p(delta):
    """Exact one-sided sign-flip over the 6 units with statistic W, 64 permutations."""
    w = plateau.weights(UNITS)
    T = float((w * delta).sum())
    hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=6):
        if float((w * np.array(signs) * delta).sum()) >= T - 1e-15:
            hits += 1
    return hits / 64.0


def noise(s0_seed0, s0_seed1, f0_seed0, f0_seed1):
    """(eps_s, eps_m) of section 4.2 from the A/A differences by unit."""
    w = np.array([metric.UNIT_W[u] for u in UNITS])
    daa_s, daa_m = s0_seed0 - s0_seed1, f0_seed0 - f0_seed1
    eps_s = max(0.019, 2 * float(np.sqrt((w ** 2 * daa_s ** 2).sum())))
    eps_m = max(0.005, 2 * float(np.sqrt((w ** 2 * daa_m ** 2).sum())))
    return eps_s, eps_m, daa_s.tolist(), daa_m.tolist()


def g0(s0, base):
    """The floor: W(S0 - base*) >= 0.04, higher on >= 5 of 6 units, positive on both sources."""
    d = s0.s - base.s
    out = {"T": W(d), "units_above": int((d > 0).sum()), "M": W_src(d, "M"), "I": W_src(d, "I")}
    out["pass"] = bool(out["T"] >= 0.04 and out["units_above"] >= 5 and out["M"] > 0 and out["I"] > 0)
    return out


def _cond3(c, s0, eps_s, units):
    d = c.s - s0.s
    eM, eI = plateau.eps_source(eps_s, units, "M"), plateau.eps_source(eps_s, units, "I")
    w = plateau.weights(units)
    worst_c = float(((w[:, None] * c.se).sum(0)).min())
    worst_0 = float(((w[:, None] * s0.se).sum(0)).min())
    return W_src(d, "M", units) >= -eM and W_src(d, "I", units) >= -eI and worst_c >= worst_0 - eps_s


def _folds_ok(c, s0, eps_s, eps_m, tie):
    """Without each unit, the count of folds that pass condition 3 and condition 1, or for a tie
    condition 5': macro in place of condition 1."""
    n = 0
    for u in UNITS:
        rest = tuple(x for x in UNITS if x != u)
        ok = _cond3(c, s0, eps_s, rest)
        if tie:
            ok = ok and W(c.f - s0.f, rest) >= eps_m
        else:
            ok = ok and W(c.s - s0.s, rest) >= eps_s
        n += bool(ok)
    return n


def judge(cands, s0, eps_s, eps_m):
    """{name: report} for {name: Nested}: the rule of 4.6 for candidates against S0."""
    rep = {}
    for name, c in cands.items():
        d, dm = c.s - s0.s, c.f - s0.f
        r = {"T": W(d), "T_macro": W(dm), "p": signflip_p(d), "p_macro": signflip_p(dm),
             "delta_u": d.tolist(), "macro_u": dm.tolist(), "M": W_src(d, "M"), "I": W_src(d, "I"),
             "T2": W(c.t2 - s0.t2), "cond3": bool(_cond3(c, s0, eps_s, UNITS))}
        r["cond3b"] = r["T2"] >= 0
        eI = plateau.eps_source(eps_s, UNITS, "I")
        r["cond3c"] = name != "C2" or r["I"] >= eI
        r["cond4"] = r["T_macro"] >= -eps_m
        r["folds5"] = int(_folds_ok(c, s0, eps_s, eps_m, False))
        r["folds5p"] = int(_folds_ok(c, s0, eps_s, eps_m, True))
        rep[name] = r
    for n, r in rep.items():
        # one threshold and p in ascending order: the stop at the first failure is p <= THRESH
        r["holm_win"], r["holm_tie"] = r["p"] <= THRESH, r["p_macro"] <= THRESH
        if r["T"] < 0:
            r["accepted"], r["route"] = False, "T < 0"
        elif r["T"] >= eps_s:
            r["accepted"] = bool(r["holm_win"] and r["cond3"] and r["cond3b"] and r["cond3c"]
                                 and r["cond4"] and r["folds5"] >= 5)
            r["route"] = "win"
        else:
            c = cands[n]
            eI = plateau.eps_source(eps_m, UNITS, "I")
            macro_i = W_src(c.f - s0.f, "I")
            macro_m = W_src(c.f - s0.f, "M")
            r["accepted"] = bool(r["T_macro"] >= eps_m and r["holm_tie"] and r["cond3"] and r["cond3b"]
                                 and (n != "C2" or macro_i >= eI) and r["folds5p"] >= 5
                                 and macro_i >= 0 and macro_m >= 0)
            r["route"] = "tie"
    return rep


SIMPLE = ("C7", "C2", "C1")             # fewer values of class Ч, then fewer columns


def best_single(rep, names, eps_s, eps_m):
    """The single with the larger T; inside the noise of T section 4.8: macro beyond eps_m, else the simpler."""
    top = max(rep[n]["T"] for n in names)
    near = [n for n in names if rep[n]["T"] >= top - eps_s]
    best_m = max(rep[n]["T_macro"] for n in near)
    near = [n for n in near if rep[n]["T_macro"] >= best_m - eps_m]
    return min(near, key=SIMPLE.index)


def combination(comb, singles, s0, rep, eps_s, eps_m):
    """P7: (name kept) among the accepted singles and their combination."""
    best = best_single(rep, list(singles), eps_s, eps_m)
    S_best = max(W(singles[n].s) for n in singles)
    has_c2 = "C2" in singles
    ok = W(comb.s) >= S_best + eps_s and W(comb.t2 - s0.t2) >= 0
    if has_c2:
        ok = ok and W_src(comb.s - s0.s, "I") >= plateau.eps_source(eps_s, UNITS, "I")
    return ("comb" if ok else best), {"S_comb": W(comb.s), "S_best": S_best, "ok": bool(ok)}
