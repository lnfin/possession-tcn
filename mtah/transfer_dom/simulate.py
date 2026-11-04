"""Player part of G: smoothing, smooth error, track breaks, a track following an opponent.

Tracks break with a constant hazard, missing stretches are sized to the copy's share. After a
contact (opponents within contact_m) a track may follow the other player for ~wrong_mean_s seconds;
where contacts cannot reach the copy's share the rest of the pieces is placed at random visible
player-frames. Every value is drawn per copy (protocol, table 1.7). Deterministic for a given seed.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

FPS = 25


@dataclass(frozen=True)
class SimParams:
    missing_share: float
    life: float                             # median life of a track, s
    off_sigma: float                        # lognormal spread of a missing piece
    wrong_share: float
    wrong_mean_s: float
    contact_m: float


def _nan_smooth(X, sigma):
    """Gaussian smoothing along axis 0 that ignores NaN (normalised convolution); NaN stays NaN."""
    if sigma <= 0:
        return X.copy()
    ok = np.isfinite(X)
    num = gaussian_filter1d(np.where(ok, X, 0.0), sigma, axis=0, mode="nearest")
    den = gaussian_filter1d(ok.astype(np.float64), sigma, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(ok, out, np.nan)


def _smooth_noise(rng, shape, sd, corr_frames):
    z = rng.standard_normal(shape)
    k = np.exp(-0.5 * (np.arange(-3 * corr_frames, 3 * corr_frames + 1) / corr_frames) ** 2)
    k /= np.sqrt((k ** 2).sum())
    flat = z.reshape(shape[0], -1)
    out = np.empty_like(flat)
    for j in range(flat.shape[1]):
        out[:, j] = np.convolve(flat[:, j], k, mode="same")
    return (out * sd).reshape(shape)


def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))


def _alternate(p_end, rng, off_median, off_sigma, max_off):
    """Bool [T] 'on' for one slot: on -> off with per-frame probability p_end, off for a lognormal
    number of frames. p_end NaN = not on the pitch."""
    T = len(p_end)
    on = np.isfinite(p_end).copy()
    u = rng.uniform(size=T)
    ev = np.flatnonzero(on & (u < np.nan_to_num(p_end)))
    t_free = -1
    for t in ev:
        if t < t_free:
            continue
        g = int(np.clip(rng.lognormal(np.log(off_median), off_sigma), 2, max_off))
        on[t + 1:t + 1 + g] = False
        t_free = t + 1 + g
    return on


def _tracks(has, rng, sp):
    """has [T, S] after the track breaks."""
    T, S = has.shape
    p = np.where(has, np.log(2.0) / (sp.life * FPS), np.nan)
    rate = np.nanmean(p)                                   # breaks per on-frame
    share = sp.missing_share
    mean_off = share / max(1 - share, 1e-3) / max(rate, 1e-9)
    off_median = max(mean_off / np.exp(sp.off_sigma ** 2 / 2), 2.0)
    out = has.copy()
    for j in range(S):
        out[:, j] = _alternate(p[:, j], rng, off_median, sp.off_sigma, 120 * FPS) & has[:, j]
    return out


def _wrong_team(P, has, team, slot_team, rng, sp):
    """(has, team): pieces where a slot carries the other team. A piece sets 1 - slot_team[i]; it
    does not toggle, so overlaps cannot give the true team back but do undershoot the share."""
    T, S = has.shape
    share = sp.wrong_share
    onsets = []
    idx = np.arange(S)
    for c in range(0, T, 4000):
        Q = np.where(has[c:c + 4000, :, None], P[c:c + 4000], np.nan).astype(np.float32)
        D = np.linalg.norm(Q[:, :, None] - Q[:, None], axis=-1)
        tm = team[c:c + 4000]
        opp = (tm[:, :, None] != tm[:, None]) & (tm[:, :, None] >= 0) & (tm[:, None] >= 0)
        close = np.where(opp, np.nan_to_num(D, nan=99.0) < sp.contact_m, False)
        close[:, idx, idx] = False
        r, i, j = np.nonzero(close)
        keep = i < j
        onsets.append(np.stack([r[keep] + c, i[keep], j[keep]], 1))
    on = np.concatenate(onsets) if onsets else np.zeros((0, 3), dtype=np.int64)
    if len(on):
        # contact onsets: the pair was not in contact on the previous frame
        key = on[:, 1] * S + on[:, 2]
        order = np.lexsort((on[:, 0], key))
        on, key = on[order], key[order]
        first = np.r_[True, (key[1:] != key[:-1]) | (on[1:, 0] - on[:-1, 0] > 1)]
        on = on[first]
    mean_d = sp.wrong_mean_s * FPS
    wanted = share * has.sum()
    q = min(1.0, wanted / max(len(on) * mean_d, 1.0))
    team = team.copy()
    has = has.copy()
    other = 1 - np.asarray(slot_team, dtype=np.int8)
    for t, i, j in on[rng.uniform(size=len(on)) < q]:
        if rng.rand() < 0.5:
            i, j = j, i
        g = int(min(rng.exponential(mean_d), 120 * FPS))
        seg = slice(t, t + g)
        team[seg, i] = np.where(has[seg, i], other[i], team[seg, i])
        has[seg, j] = False
    if q >= 1.0 and wanted > len(on) * mean_d:
        rows, slots = np.nonzero(has)
        n_more = int(np.ceil((wanted - len(on) * mean_d) / mean_d))
        for k in rng.randint(len(rows), size=n_more):
            t, i = rows[k], slots[k]
            g = int(min(rng.exponential(mean_d), 120 * FPS))
            seg = slice(t, t + g)
            team[seg, i] = np.where(has[seg, i], other[i], team[seg, i])
    return has, team
