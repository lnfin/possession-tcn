"""G: an automatic-tracker copy of a clean open match, 46 slots per copy (schema 1.7).

    players   transfer_dom/simulate.py; every piece of a player with one team (between breaks) goes
              into the first free slot of that team
    ball      a dead ball where the clean ball vanishes once play stops (Metrica: corrupt._dead_ball),
              Gaussian smoothing, error, lag, then the gaps: on a visible frame the ball goes with
              probability sigmoid(b + a_c min(n_c, 6) / 3 + a_h min(z+ / 3, 2) + a_s speed / 10), the
              gap's length is drawn at once; b is calibrated by bisection to the copy's hidden share

Labels are not touched here; team swap and mirror are in analysis/clean/bank.py.
Deterministic for a given seed.
"""

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from ..external import corrupt
from ..external.corrupt import DeadBallParams
from ..transfer_dom import hazards as hz
from ..transfer_dom import simulate as sim1

FPS = 25
B_LO, B_HI = -14.0, 0.0
B_ITERS = 14
B_TOL = 0.02


@dataclass(frozen=True)
class SimParams2:
    seed: int
    h: float                                # hidden share of period frames
    a_c: float
    a_h: float
    a_s: float
    gap_median_s: float
    gap_sigma: float
    ball_sigma: float                       # Gaussian smoothing of the true ball, frames
    ball_err_m: float                       # smooth position error (sd per axis), m
    ball_lag: int                           # frames behind the players
    player_sigma: float
    player_err_m: float
    missing_share: float
    life: float
    off_sigma: float
    wrong_share: float
    wrong_mean_s: float
    contact_m: float
    crowd_r_m: float
    dead: DeadBallParams

    def v1(self):
        return sim1.SimParams(missing_share=self.missing_share, life=self.life, off_sigma=self.off_sigma,
                              wrong_share=self.wrong_share, wrong_mean_s=self.wrong_mean_s,
                              contact_m=self.contact_m)


def period_of(game):
    period = np.zeros(len(game["ball"]), dtype=np.int8)
    for per, first, last in game["periods"]:
        period[first:last + 1] = per
    return period


def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))


def clean_ball(game, ref_codes, log, dead_params, rng_g):
    """[T, 2] metres: the clean ball, with a dead ball added where it vanishes once dead."""
    L, W = game["pitch_m"]
    Bm = np.asarray(game["ball"], dtype=np.float64) * np.array([L, W])
    dead = ref_codes == 2
    if dead.any() and np.isfinite(Bm[dead, 0]).mean() < 0.5:
        add = corrupt._dead_ball(rng_g, Bm, ref_codes, log, dead_params, game["fps"])
        Bm = np.where(np.isfinite(add), add, Bm)
    return Bm


# ------------------------------------------------------------------------------------ ball gaps

def _hide(x, b, u, length, seen, period):
    """Bool [T]: frames inside gaps for this b. x [T] the covariate part of the logit."""
    hide = np.zeros(len(x), dtype=bool)
    p = 1.0 / (1.0 + np.exp(-(b + x)))
    hits = np.flatnonzero(seen & (u < p) & (period > 0))
    for per in (1, 2):
        rows = np.flatnonzero(period == per)
        if not len(rows):
            continue
        a, z = rows[0], rows[-1] + 1
        h = hits[(hits >= a) & (hits < z)].tolist()
        i, n = 0, len(h)
        while i < n:
            s = h[i]
            stop = min(s + 1 + int(length[s]), z)
            hide[s + 1:stop] = True
            i = bisect_left(h, stop, i + 1)
    return hide


def gap_mask(B, P, has, ball_z, period, sp, rng):
    """(hide [T], info): the gaps of step 4 with b from step 5; the numbers drawn once, for every b."""
    n_c, speed = hz.ball_state(P, has, B, sp.crowd_r_m)
    zp = np.zeros(len(B)) if ball_z is None else np.maximum(np.nan_to_num(ball_z, nan=0.0), 0.0)
    x = sp.a_c * np.minimum(n_c, 6) / 3.0 + sp.a_h * np.minimum(zp / 3.0, 2.0) + sp.a_s * speed / 10.0
    u = rng.uniform(size=len(B))
    z = rng.standard_normal(len(B))
    length = np.clip(np.exp(np.log(sp.gap_median_s * FPS) + sp.gap_sigma * z), 1, 600 * FPS).astype(np.int64)
    seen = np.isfinite(B[:, 0])
    ok = period > 0
    base = float((~seen)[ok].mean())
    target = max(sp.h, base)

    def share(mask):
        return float((mask | ~seen)[ok].mean())

    if target == base:
        b = B_LO
        best = (abs(share(_hide(x, b, u, length, seen, period)) - target), b)
    else:
        lo, hi, best = B_LO, B_HI, None
        for _ in range(B_ITERS):
            b = 0.5 * (lo + hi)
            s = share(_hide(x, b, u, length, seen, period))
            if best is None or abs(s - target) < best[0]:
                best = (abs(s - target), b)
            if abs(s - target) <= B_TOL:
                break
            if s < target:
                lo = b
            else:
                hi = b
    hide = _hide(x, best[1], u, length, seen, period)
    return hide, {"b": best[1], "h_star": target, "base": base, "hidden": share(hide), "hidden_err": best[0]}


# --------------------------------------------------------------------------------------- the copy

def _players(game, sp, rng):
    """(P smoothed + error [T, S, 2] m, has, team [T, S], wrong share)."""
    L, W = game["pitch_m"]
    P0 = np.asarray(game["players"], dtype=np.float64) * np.array([L, W])
    has = np.isfinite(P0[..., 0])
    slot_team = np.asarray(game["slot_team"], dtype=np.int8)
    team = np.where(has, slot_team[None, :], -1).astype(np.int8)
    v1 = sp.v1()
    Ps = sim1._nan_smooth(P0, sp.player_sigma)
    P = Ps + sim1._smooth_noise(rng, Ps.shape, sp.player_err_m, 12)
    has = sim1._tracks(has, rng, v1)
    has, team = sim1._wrong_team(P, has, team, slot_team, rng, v1)
    team = np.where(has, team, -1).astype(np.int8)
    wrong = float((team[has] != np.broadcast_to(slot_team, has.shape)[has]).mean())
    return P, has, team, wrong


def layout(game, P, has, team):
    """(players [T, 46, 2] normalised, slot_team [46])."""
    L, W = game["pitch_m"]
    T, S, _ = P.shape
    pieces = []
    for j in range(S):
        for k in (0, 1):
            for a, b in _runs(has[:, j] & (team[:, j] == k)):
                pieces.append((int(a), int(b), j, k))
    out = np.full((T, corrupt.N_SLOTS, 2), np.nan)
    busy = np.zeros(corrupt.N_SLOTS, dtype=np.int64)
    for a, b, j, k in sorted(pieces):
        pool = corrupt.OUTFIELD[k]
        free = pool[busy[pool] <= a]
        if not len(free):
            continue                                    # the tracker is out of slots: piece lost
        slot = int(free[0])
        out[a:b, slot] = P[a:b, j]
        busy[slot] = b
    return out / np.array([L, W]), corrupt.SLOT_TEAM


def to_atd(game, ref_codes, log, sp):
    """(players [N, 46, 2], ball [N, 2], slot_team [46], info) -- corrupt.to_atd's contract."""
    rng = np.random.RandomState(sp.seed)
    rng_g = np.random.default_rng(sp.seed)
    period = period_of(game)
    L, W = game["pitch_m"]
    P, has, team, wrong = _players(game, sp, rng)

    B = clean_ball(game, ref_codes, log, sp.dead, rng_g)
    B = sim1._nan_smooth(B, sp.ball_sigma)
    B += sim1._smooth_noise(rng, B.shape, sp.ball_err_m, 12)
    if sp.ball_lag:
        B = np.r_[np.full((sp.ball_lag, 2), np.nan), B[:-sp.ball_lag]]
    hide, info = gap_mask(B, P, has, game.get("ball_z"), period, sp, rng)
    hide |= ~np.isfinite(B[:, 0])
    live = np.isin(ref_codes, (0, 1))
    B[hide] = np.nan

    P = np.where(has[..., None], P, np.nan)
    players, slot_team = layout(game, P, has, team)
    info.update(wrong_share_actual=wrong, gap_live=float(hide[live].mean()),
                players_visible=float(np.isfinite(players[..., 0])[period > 0].sum(1).mean()))
    return players.astype(np.float32), (B / np.array([L, W])).astype(np.float32), slot_team, info
