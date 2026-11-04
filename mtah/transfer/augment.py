"""The fallback generator: independent random corruption of a clean match.

Each corruption has its strength drawn per copy from the priors of the protocol, table 1.7. Labels
are never touched. Deterministic for a given seed.
"""

from dataclasses import dataclass

import numpy as np

from ..external.corrupt import DeadBallParams

FPS = 25


@dataclass(frozen=True)
class AugParams:
    h: float                # share of live frames with the ball hidden, with u
    u: float
    dead_missing: float     # chance that a stoppage loses the ball
    gap_median_s: float     # median ball gap length
    gap_sigma: float
    vis_sigma: float        # lognormal spread of a visible run
    ball_noise_m: float     # smooth ball position error (sd)
    ball_lag: int           # frames
    player_noise_m: float
    player_missing: float   # share of player-frames dropped, in pieces
    on_mean: float          # mean visible piece of a track, s
    off_sigma: float
    wrong_team: float       # share of player-frames given to the other team
    wrong_mean_s: float
    dead: DeadBallParams


def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))


def _smooth_noise(rng, shape, sd, corr_frames):
    """Gaussian noise along axis 0, smoothed over ~corr_frames, rescaled to sd."""
    z = rng.standard_normal(shape)
    k = np.exp(-0.5 * (np.arange(-3 * corr_frames, 3 * corr_frames + 1) / corr_frames) ** 2)
    k /= np.sqrt((k ** 2).sum())
    flat = z.reshape(shape[0], -1)
    out = np.empty_like(flat)
    for j in range(flat.shape[1]):
        out[:, j] = np.convolve(flat[:, j], k, mode="same")
    return (out * sd).reshape(shape)


def _ball_gaps(rng, live, dead, a):
    """Boolean [T]: frames where the ball is hidden."""
    T = len(live)
    hide = np.zeros(T, dtype=bool)
    share = min(a.h * a.u, 1.0)
    med = a.gap_median_s * FPS
    # alternating renewal process: gaps lognormal around med, visible runs sized for the share
    vis_med = med * (1 - share) / max(share, 1e-3)
    t = int(rng.uniform(0, max(vis_med, 1.0)))
    while t < T:
        g = int(np.clip(rng.lognormal(np.log(med), a.gap_sigma), 2 * FPS, 150 * FPS))
        hide[t:t + g] = True
        t += g + int(np.clip(rng.lognormal(np.log(max(vis_med, 1.0)), a.vis_sigma), FPS, 300 * FPS))
    hide &= live
    for s, e in _runs(dead):
        if rng.rand() < a.dead_missing:
            s2 = s + int(rng.uniform(0, 2) * FPS)
            e2 = e + int(rng.uniform(-1.5, 1.0) * FPS)
            hide[max(s2, 0):max(min(e2, T), 0)] = True
    return hide


def corrupt(m, seed, a):
    """(players [T, S, 2], team [T, S], ball [T, 2], labels [T]) -- a copy of Match m."""
    rng = np.random.RandomState(seed)
    T, S, _ = m.players.shape
    P = m.players.astype(np.float64).copy()
    B = m.ball.astype(np.float64).copy()
    team = np.broadcast_to(m.slot_team[None, :].astype(np.int8), (T, S)).copy()
    team[~np.isfinite(P[..., 0])] = -1

    # players: smooth position error, dropped pieces of track
    P += _smooth_noise(rng, P.shape, a.player_noise_m, 12)
    drop_share = a.player_missing
    on_mean = a.on_mean * FPS
    for j in range(S):
        t = 0
        off_med = on_mean * drop_share / max(1 - drop_share, 1e-3)
        while t < T:
            t += int(rng.exponential(on_mean))
            g = int(np.clip(rng.lognormal(np.log(max(off_med, 5)), a.off_sigma), 5, 120 * FPS))
            team[t:t + g, j] = -1
            t += g
    # wrong team: pieces where a track follows the other team
    mean_w = a.wrong_mean_s * FPS
    n_pieces = int(a.wrong_team * T * S / mean_w)
    for _ in range(n_pieces):
        j = rng.randint(S)
        s = rng.randint(T)
        g = int(min(rng.exponential(mean_w), 120 * FPS))
        seg = team[s:s + g, j]
        team[s:s + g, j] = np.where(seg >= 0, 1 - seg, seg)

    # ball: lag, smooth error, gaps in live play and stoppages
    if a.ball_lag:
        B = np.r_[np.full((a.ball_lag, 2), np.nan), B[:-a.ball_lag]]
    B += _smooth_noise(rng, B.shape, a.ball_noise_m, 8)
    live = np.isin(m.labels, ["H", "A"])
    dead = m.labels == "D"
    B[_ball_gaps(rng, live, dead, a)] = np.nan
    return P.astype(np.float32), team, B.astype(np.float32), m.labels.copy()
