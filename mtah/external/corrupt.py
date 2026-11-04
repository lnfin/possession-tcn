"""T2: turn clean tracking into an automatic-tracker copy.

Layout: players [N, 46, 2] (x, y normalised), ball [N, 2], slot_team [46] -- slots 0-23 home, 24-45
away. Every parameter is drawn per copy from the priors of the protocol, table 1.7; nothing here
has a default. Ball gaps start at random times, their rate depends on live / dead; player tracks
break and resume in another slot, a slot may follow a player of the other team.

Deterministic for a given seed.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

N_SLOTS = 46
SLOT_TEAM = np.r_[np.zeros(24), np.ones(22)].astype(np.int8)
OUTFIELD = {k: np.flatnonzero(SLOT_TEAM == k) for k in (0, 1)}


@dataclass(frozen=True)
class DeadBallParams:
    out_seen: float                # the ball is seen going on over the line
    roll_seen: float               # ... or rolling on after a whistle
    out_median_s: float            # ... this long
    out_sigma: float
    out_tau_s: float               # its speed decays with this time constant
    carry_p: float                 # the ball is seen being brought to the spot
    carry_speed: float             # m/s
    placed_p: float                # a restart's ball is seen still at the spot beforehand ...
    placed_median_s: float         # ... for about this long
    placed_sigma: float


@dataclass(frozen=True)
class CorruptParams:
    seed: int
    level: str                     # "atd"
    # ball
    h: float                       # share of period frames without the ball, through gap_live / gap_dead
    u: float                       # 1: gaps do not depend on the label; > 1: the dead ball is seen better
    gap_median_s: float
    gap_sigma: float
    ball_lag: int
    ball_smooth: int
    ball_noise_m: float            # per axis, slowly varying (AR, then a ball_noise_smooth-frame mean)
    ball_tau_s: float
    ball_noise_smooth: int
    # players
    pos_noise_m: float             # per axis
    pos_tau_s: float
    missing: float
    drop_median_s: float
    drop_sigma: float
    life_median_s: float
    life_sigma: float
    wrong_team: float
    wrong_median_s: float
    wrong_sigma: float
    dead: DeadBallParams


def _lognormal(rng, median, sigma, size=None):
    return np.exp(np.log(median) + sigma * rng.standard_normal(size))


def _ar_noise(rng, shape, sigma, tau_frames):
    phi = np.exp(-1.0 / tau_frames)
    z = rng.standard_normal(shape) * sigma * np.sqrt(1 - phi ** 2)
    return lfilter([1.0], [1.0, -phi], z, axis=0)


def _runs(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


# ------------------------------------------------------------------------------------- players

def _tracks(rng, on, p, fps):
    """[(a, b, wrong_team)] visible pieces of one player's presence mask, each its own track."""
    n = len(on)
    vis = on.copy()
    mean_drop = p.drop_median_s * fps * np.exp(p.drop_sigma ** 2 / 2)
    if p.missing > 0:
        t = 0
        while t < n:
            t += int(rng.exponential(mean_drop * (1 - p.missing) / p.missing))
            d = int(_lognormal(rng, p.drop_median_s * fps, p.drop_sigma))
            vis[t:t + d] = False
            t += d
    out = []
    mean_wrong = p.wrong_median_s * fps * np.exp(p.wrong_sigma ** 2 / 2)
    for a, b in _runs(vis):
        cuts = [a]
        while True:
            nxt = cuts[-1] + max(25, int(_lognormal(rng, p.life_median_s * fps, p.life_sigma)))
            if nxt >= b:
                break
            cuts.append(nxt)
        cuts.append(b)
        for s, e in zip(cuts[:-1], cuts[1:]):
            # carve wrong-team pieces out of the track
            t = s
            while p.wrong_team > 0:
                on_at = t + int(rng.exponential(mean_wrong * (1 - p.wrong_team) / p.wrong_team))
                if on_at >= e:
                    out.append((t, e, False))
                    break
                off_at = min(e, on_at + max(5, int(_lognormal(rng, p.wrong_median_s * fps, p.wrong_sigma))))
                if on_at > t:
                    out.append((t, on_at, False))
                out.append((on_at, off_at, True))
                t = off_at
                if t >= e:
                    break
            else:
                out.append((s, e, False))
    return out


def players_to_atd(game, p, rng):
    P = np.asarray(game["players"], dtype=np.float64)
    team = game["slot_team"]
    L, W = game["pitch_m"]
    n, s0, _ = P.shape
    out = np.full((n, N_SLOTS, 2), np.nan)
    noise = _ar_noise(rng, (n, s0, 2), p.pos_noise_m, p.pos_tau_s * game["fps"])
    Pm = P * np.array([L, W]) + noise
    busy = np.zeros(N_SLOTS, dtype=np.int64)          # frame until which each slot is taken
    pieces = []
    for j in range(s0):
        on = np.isfinite(P[:, j, 0])
        for a, b, wrong in _tracks(rng, on, p, game["fps"]):
            pieces.append((a, b, j, int(team[j]) ^ int(wrong)))
    for a, b, j, k in sorted(pieces):
        pool = OUTFIELD[k]
        free = pool[busy[pool] <= a]
        if not len(free):
            continue                                    # tracker out of slots: the piece is lost
        slot = int(free[0])
        out[a:b, slot] = Pm[a:b, j]
        busy[slot] = b
    out /= np.array([L, W])
    return out.astype(np.float32), SLOT_TEAM


# ---------------------------------------------------------------------------------------- ball

def _gap_mask(rng, dead, base_missing, graded, gap_live, gap_dead, p, fps):
    """Boolean missing mask: renewal process whose gap onset rate depends on live / dead."""
    n = len(dead)
    h = {False: 1.0 / (30 * fps), True: 1.0 / (30 * fps)}
    best = None
    dead_list = dead.tolist()
    for _ in range(12):
        r = np.random.default_rng(rng.integers(1 << 31))
        miss = base_missing.copy()
        u = r.random(n).tolist()
        t = 0
        while t < n:
            if u[t] < h[dead_list[t]]:
                d = int(np.clip(_lognormal(r, p.gap_median_s * fps, p.gap_sigma), 1, 600 * fps))
                miss[t:t + d] = True
                t += d
            else:
                t += 1
        live_share = miss[graded & ~dead].mean()
        dead_share = miss[graded & dead].mean()
        err = abs(live_share - gap_live) + abs(dead_share - gap_dead)
        if best is None or err < best[0]:
            best = (err, miss, live_share, dead_share)
        h[False] *= gap_live / max(live_share, 1e-3)
        h[True] *= max(gap_dead - base_missing[graded & dead].mean(), 1e-3) / \
            max(dead_share - base_missing[graded & dead].mean(), 1e-3)
    return best[1:]


def _dead_ball(rng, Bm, ref_codes, log, d, fps):
    """The ball a tracker sees while it is dead, which an event-corrected clean ball lacks.

    Per dead stretch: the ball going on over the line (or rolling on after a whistle) for a moment,
    then being brought to the restart spot, then lying there until the kick.
    Returns (positions [N, 2] metres, NaN where nothing is added).
    """
    n = len(Bm)
    add = np.full((n, 2), np.nan)
    kind_at = {e["restart"]: e["kind"] for e in log}
    for a, r in _runs(ref_codes == 2):
        if r >= n or ref_codes[r] < 0 or r - a < fps:
            continue
        seen = np.flatnonzero(np.isfinite(Bm[max(0, a - 2 * fps):a + 1, 0]))
        after = np.flatnonzero(np.isfinite(Bm[r:r + 2 * fps, 0]))
        if not len(after):
            continue
        spot = Bm[r + after[0]]
        t = a
        if len(seen) >= 3:
            i = max(0, a - 2 * fps) + seen[-1]
            p0, v0 = Bm[i], (Bm[i] - Bm[max(0, i - 3)]) / 3.0 * fps
            kind = kind_at.get(r, "out")
            go_on = d.out_seen if kind in ("out", "goal") else d.roll_seen
            if rng.random() < go_on:
                k = min(r - a, max(3, int(_lognormal(rng, d.out_median_s * fps, d.out_sigma))))
                tt = np.arange(1, k + 1) / fps
                decay = d.out_tau_s * (1 - np.exp(-tt / d.out_tau_s))
                add[a:a + k] = p0 + v0[None] * decay[:, None]
                t = a + k
        if rng.random() < d.carry_p and r - t > 2 * fps:
            c0 = t + int(rng.uniform(0, 1) * (r - t))
            start = (add[t - 1] if np.isfinite(add[t - 1, 0]) else spot + rng.normal(0, 8, 2))
            k = int(np.linalg.norm(spot - start) / d.carry_speed * fps) + 1
            k = min(k, r - c0)
            w = np.linspace(0, 1, k)[:, None]
            add[c0:c0 + k] = start + w * (spot - start)
            if rng.random() < d.placed_p:
                add[c0 + k:r] = spot
        elif rng.random() < d.placed_p:
            k = min(r - t, int(_lognormal(rng, d.placed_median_s * fps, d.placed_sigma)))
            add[r - k:r] = spot
    return add


def ball_to_atd(game, ref_codes, log, p, rng):
    B = np.asarray(game["ball"], dtype=np.float64)
    L, W = game["pitch_m"]
    n = len(B)
    fps = game["fps"]
    Bm = B * np.array([L, W])
    if np.isfinite(Bm[ref_codes == 2, 0]).mean() < 0.5:
        # an event-corrected ball (Metrica) vanishes when it goes dead; a raw one (DFL) does not
        add = _dead_ball(rng, Bm, ref_codes, log, p.dead, fps)
        Bm = np.where(np.isfinite(add), add, Bm)
    Bm = np.r_[np.full((p.ball_lag, 2), np.nan), Bm[:n - p.ball_lag]]
    noise = _ar_noise(rng, (n, 2), p.ball_noise_m, p.ball_tau_s * fps)
    k = np.ones(p.ball_noise_smooth) / p.ball_noise_smooth
    Bm += np.stack([np.convolve(noise[:, c], k, "same") for c in range(2)], 1)
    if p.ball_smooth > 1:
        k = np.ones(p.ball_smooth) / p.ball_smooth
        Bm = np.stack([np.convolve(Bm[:, c], k, "same") for c in range(2)], 1)
    graded = ref_codes >= 0
    dead = ref_codes == 2
    share_dead = dead[graded].mean()
    gap_live = min(p.h * p.u, 1.0)
    gap_dead = float(np.clip((p.h - gap_live * (1 - share_dead)) / share_dead, 0.0, 1.0))
    miss, live_share, dead_share = _gap_mask(rng, dead, ~np.isfinite(Bm[:, 0]), graded, gap_live,
                                             gap_dead, p, fps)
    Bm[miss] = np.nan
    info = {"ball_missing_live": float(live_share), "ball_missing_dead": float(dead_share),
            "gap_live": gap_live, "gap_dead": gap_dead}
    return (Bm / np.array([L, W])).astype(np.float32), info


def to_atd(game, ref_codes, log, p):
    """(players [N,46,2], ball [N,2], slot_team [46], info).

    ref_codes / log (reference.labels) say where the ball is dead: the rate of gaps depends on it.
    """
    rng = np.random.default_rng(p.seed)
    players, slot_team = players_to_atd(game, p, rng)
    ball, info = ball_to_atd(game, ref_codes, log, p, rng)
    info["players_visible"] = float(np.isfinite(players[..., 0])[ref_codes >= 0].sum(1).mean())
    return players, ball, slot_team, info
