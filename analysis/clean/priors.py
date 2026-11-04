"""The priors of table 1.7, the order of their stream, the widened T1, the stress of T3.

A prior is ("U", a, b), ("LU", a, b) log-uniform, ("I", a, b) an integer in [a, b], ("ODD", a, b) an odd
integer in [a, b]. draw(kind, c) samples a copy's parameters from default_rng([c, 1]) in the fixed order;
an overridden field is not drawn.
"""

import hashlib
import json
import math

import numpy as np

from mtah.external.corrupt import CorruptParams, DeadBallParams
from mtah.transfer.augment import AugParams
from mtah.transfer_unified.simulate2 import SimParams2

H_ = ("U", 0.16, 0.66)
SHARE = ("U", 0.0, 1.0)               # Д
DUR = ("LU", 1.0, 60.0)               # ДЛ
ERR = ("LU", 0.05, 1.0)               # ОШ
SPREAD = ("U", 0.0, 2.0)              # РАЗ
SLOPE = ("U", 0.0, 2.0)               # НК
HALF = ("U", 0.0, 0.5)
LAG = ("I", 0, 5)
ODD = ("ODD", 1, 31)
U_ = ("LU", 0.5, 2.0)

DEAD = (("out_seen", SHARE), ("roll_seen", SHARE), ("out_median_s", DUR), ("out_sigma", SPREAD),
        ("out_tau_s", ("U", 0.2, 4.0)), ("carry_p", SHARE), ("carry_speed", ("U", 1.0, 12.0)),
        ("placed_p", SHARE), ("placed_median_s", DUR), ("placed_sigma", SPREAD))
G = (("h", H_), ("a_c", SLOPE), ("a_h", SLOPE), ("a_s", SLOPE), ("gap_median_s", DUR),
     ("gap_sigma", SPREAD), ("ball_sigma", ("LU", 0.5, 16.0)), ("ball_err_m", ERR), ("ball_lag", LAG),
     ("player_sigma", ("LU", 0.5, 8.0)), ("player_err_m", ERR), ("missing_share", HALF), ("life", DUR),
     ("off_sigma", SPREAD), ("wrong_share", HALF), ("wrong_mean_s", DUR),
     ("contact_m", ("LU", 0.5, 3.0)), ("crowd_r_m", ("U", 1.0, 6.0)))
T2 = (("h", H_), ("u", U_), ("gap_median_s", DUR), ("gap_sigma", SPREAD), ("ball_lag", LAG),
      ("ball_smooth", ODD), ("ball_noise_m", ERR), ("ball_tau_s", DUR), ("ball_noise_smooth", ODD),
      ("pos_noise_m", ERR), ("pos_tau_s", DUR), ("missing", HALF), ("drop_median_s", DUR),
      ("drop_sigma", SPREAD), ("life_median_s", DUR), ("life_sigma", SPREAD), ("wrong_team", HALF),
      ("wrong_median_s", DUR), ("wrong_sigma", SPREAD))
AUG = (("h", H_), ("u", U_), ("dead_missing", SHARE), ("gap_median_s", DUR), ("gap_sigma", SPREAD),
       ("vis_sigma", SPREAD), ("ball_noise_m", ERR), ("ball_lag", LAG), ("player_noise_m", ERR),
       ("player_missing", HALF), ("on_mean", DUR), ("off_sigma", SPREAD), ("wrong_team", HALF),
       ("wrong_mean_s", DUR))

# area of each field for T1's clip, section 1.7
DOMAIN = {"h": (0.0, 1.0), "missing_share": (0.0, 0.5), "wrong_share": (0.0, 0.5), "player_missing": (0.0, 0.5),
          "wrong_team": (0.0, 0.5), "missing": (0.0, 0.5), "dead_missing": (0.0, 1.0), "out_seen": (0.0, 1.0),
          "roll_seen": (0.0, 1.0), "carry_p": (0.0, 1.0), "placed_p": (0.0, 1.0),
          "carry_speed": (0.5, math.inf), "out_tau_s": (0.05, math.inf)}
WIDEN = 0.625


def widen(name, prior):
    """The T1 prior of a field: mid +- 0.625 (b - a), in the log for LU; clipped; integers one up."""
    kind, a, b = prior
    if kind == "I":
        return (kind, a, b + 1)
    if kind == "ODD":
        return prior
    lo, hi = DOMAIN.get(name, (0.0, math.inf))
    if kind == "LU":
        la, lb = math.log(a), math.log(b)
        mid, half = 0.5 * (la + lb), WIDEN * (lb - la)
        return (kind, max(math.exp(mid - half), lo), min(math.exp(mid + half), hi))
    mid, half = 0.5 * (a + b), WIDEN * (b - a)
    return (kind, max(mid - half, lo), min(mid + half, hi))


def _sample(r, prior):
    kind, a, b = prior
    if kind == "U":
        return float(r.uniform(a, b))
    if kind == "LU":
        return float(np.exp(r.uniform(np.log(a), np.log(b))))
    if kind == "I":
        return int(r.integers(a, b + 1))
    return int(a + 2 * r.integers(0, (b - a) // 2 + 1))      # ODD


def _draw_fields(r, fields, wide, over):
    out = {}
    for name, prior in fields:
        if name in over:
            out[name] = over[name]
        else:
            out[name] = _sample(r, widen(name, prior) if wide else prior)
    return out


def t1_bound(fields, name, side):
    prior = dict(fields)[name]
    return widen(name, prior)[1 if side == "lo" else 2]


T3A = {"h": t1_bound(G, "h", "hi"), "ball_sigma": t1_bound(G, "ball_sigma", "hi"),
       "player_sigma": t1_bound(G, "player_sigma", "hi"), "ball_err_m": t1_bound(G, "ball_err_m", "hi"),
       "player_err_m": t1_bound(G, "player_err_m", "hi"), "missing_share": t1_bound(G, "missing_share", "hi"),
       "wrong_share": 0.375, "life": t1_bound(G, "life", "lo")}
T3B = {"ball_smooth": 31, "ball_noise_smooth": 31, "h": t1_bound(T2, "h", "hi"),
       "pos_noise_m": t1_bound(T2, "pos_noise_m", "hi"), "ball_noise_m": t1_bound(T2, "ball_noise_m", "hi"),
       "ball_tau_s": t1_bound(T2, "ball_tau_s", "hi"), "pos_tau_s": t1_bound(T2, "pos_tau_s", "hi"),
       "missing": t1_bound(T2, "missing", "hi"), "wrong_team": 0.375,
       "life_median_s": t1_bound(T2, "life_median_s", "lo"), "u": 1.0}


def draw(kind, c):
    """Parameters of the copy with seed c. kind: G, T1, T2, T3a, T3b, AUG, AUG1 (fallback G, T1)."""
    r = np.random.default_rng([c, 1])
    if kind in ("G", "T1", "T3a"):
        wide = kind != "G"
        own = _draw_fields(r, G, wide, T3A if kind == "T3a" else {})
        dead = DeadBallParams(**_draw_fields(r, DEAD, wide, {}))
        return SimParams2(seed=c, dead=dead, **own)
    if kind in ("T2", "T3b"):
        own = _draw_fields(r, T2, False, T3B if kind == "T3b" else {})
        dead = DeadBallParams(**_draw_fields(r, DEAD, False, {}))
        return CorruptParams(seed=c, level="atd", dead=dead, **own)
    wide = kind == "AUG1"
    own = _draw_fields(r, AUG, wide, {})
    dead = DeadBallParams(**_draw_fields(r, DEAD, wide, {}))
    return AugParams(dead=dead, **own)


def table_md5():
    """md5 of the table, for the registry."""
    body = {"DEAD": DEAD, "G": G, "T2": T2, "AUG": AUG, "T3A": T3A, "T3B": T3B, "WIDEN": WIDEN}
    return hashlib.md5(json.dumps(body, sort_keys=True).encode()).hexdigest()
