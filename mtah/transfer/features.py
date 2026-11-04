"""Per-frame features that any tracking feed has: team-level, no player identity (version 2).

Input: players [T, S, 2] in metres (origin at a pitch corner, NaN = not tracked), the team of
every player on every frame [T, S] (0 home, 1 away, -1 none -- per frame, so that a track that
swaps teams can be represented), the raw ball [T, 2] (NaN = not tracked), the period of each frame
and the pitch size. The same function serves the copies of the open matches and the input of run.py.

Coordinates are turned so that +x is the home team's attacking direction in each period, read from
the data (the home block's mean x is behind the away block's), and scaled by the pitch size. Nothing
depends on a slot keeping its player.

The order of the columns is the list of the protocol, table 1.6. Control is computed for a tuple of
radii: `radii` are the three rungs r0 * (1, 2, 4) in the model, all seven values of the grid in the
bank (ALL_RADII); columns of a radius carry the radius in their name in the bank, the rung in the model.
"""

import warnings

import numpy as np

from .. import ball as ballmod

FPS = 25
GAP_CAP = 30.0 * FPS
FEATURE_VERSION = 2
PLAYER_SPEED_CAP = 12.0          # m/s, law of the data: identity switches jump
ALL_RADII = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
RUNG = (1, 2, 4)
SIDES = ("H", "A")
SHAPE_STATS = ("n", "cx", "cy", "sx", "sy", "x10", "x90", "y10", "y90", "speed", "vx", "own_half",
               "len", "wid")
SHAPE_PAIR = ("gap_cx", "front_back", "back_front", "speed_diff", "n_centre")
CTRL = ("ctrl_H", "ctrl_A", "ctrl_last", "ctrl_next", "ctrl_since", "ctrl_until")


def _head():
    out = ["ball_present", "ball_x", "ball_y", "ball_absy", "ball_speed", "ball_out_dist"]
    for side in SIDES:
        out += [k + "_" + side for k in ("rel", "logd1", "logd2", "n3", "n5", "n10")]
    return out + ["d_diff"]


def _tail():
    out = ["ball_since", "ball_until", "ball_last_x", "ball_next_x", "ball_last_out", "ball_next_out",
           "ball_last_absy", "ball_next_absy"]
    for w in (25, 75):
        for side in ("h", "a"):
            out += ["{}_{}@{}".format(side, k, w) for k in SHAPE_STATS]
        out += ["{}@{}".format(k, w) for k in SHAPE_PAIR]
    return out + ["pspeed1", "pspeed@75"]


def names_rungs():
    """The 113 model columns, in the order of table 1.6."""
    return _head() + ["{}@{}".format(k, r) for r in range(len(RUNG)) for k in CTRL] + _tail()


def names_bank(radii=ALL_RADII):
    """The bank columns: as the model's, with one control group per radius (137 for all seven)."""
    return _head() + ["{}@r{}".format(k, r) for r in radii for k in CTRL] + _tail()


def bank_columns(r0):
    """Indices into names_bank() of the 113 model columns of this r0, in model order."""
    bank = {k: i for i, k in enumerate(names_bank())}
    out = []
    for name in names_rungs():
        key, _, tail = name.partition("@")
        if key in CTRL:
            name = "{}@r{}".format(key, r0 * RUNG[int(tail)])
        out.append(bank[name])
    return out


def home_attacks_right(players, team, period):
    """{period: True if home attacks +x}, from the blocks' mean x over the period."""
    out = {}
    X = players[..., 0].astype(np.float64)
    for per in (1, 2):
        rows = period == per
        if not rows.any():
            continue
        xs = X[rows]
        tm = team[rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mh = np.nanmean(np.where(tm == 0, xs, np.nan))
            ma = np.nanmean(np.where(tm == 1, xs, np.nan))
        out[per] = bool(mh < ma)
    return out


def _rolling_mean(x, w):
    """Centred rolling mean over axis 0 ignoring NaN; NaN where the window has no data."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    k = np.ones(w)
    num = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, v)
    den = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, ok.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _vel(X):
    """Central difference over +-2 frames, per second; NaN where either end is missing."""
    V = np.full_like(X, np.nan)
    V[2:-2] = (X[4:] - X[:-4]) * (FPS / 4.0)
    return V


def _nanpct(x, q):
    """Row-wise q-th percentile (linear interpolation) ignoring NaN; NaN on empty rows.
    np.nanpercentile on [T, S] is the slowest step otherwise."""
    srt = np.sort(x, axis=1)                       # NaN sorts last
    n = np.isfinite(x).sum(1)
    pos = (q / 100.0) * np.maximum(n - 1, 0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
    r = np.arange(len(x))
    v = srt[r, lo] + (pos - lo) * (srt[r, hi] - srt[r, lo])
    return np.where(n > 0, v, np.nan)


def _since(flag, cap=GAP_CAP):
    """(frames since flag was last True, capped / cap; index of that frame; was there one)."""
    idx = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, idx, -1))
    dt = np.where(last >= 0, idx - last, cap)
    return np.minimum(dt, cap) / cap, np.maximum(last, 0), last >= 0


def _until(flag, cap=GAP_CAP):
    d, nxt_r, ok_r = _since(flag[::-1], cap)
    n = len(flag)
    return d[::-1], (n - 1 - nxt_r)[::-1], ok_r[::-1]


def _control(cols, suffix, r, present, d1):
    ctrl = np.full(len(present), -1, dtype=np.int8)
    ctrl[present & (d1["H"] <= r) & (d1["A"] > d1["H"])] = 0
    ctrl[present & (d1["A"] <= r) & (d1["H"] > d1["A"])] = 1
    cols["ctrl_H@" + suffix] = (ctrl == 0) * 1.0
    cols["ctrl_A@" + suffix] = (ctrl == 1) * 1.0
    hasc = ctrl >= 0
    s, last, ok = _since(hasc)
    u, nxt, ok2 = _until(hasc)
    cols["ctrl_last@" + suffix] = np.where(ok, np.where(ctrl[last] == 1, 1.0, -1.0), 0.0)
    cols["ctrl_next@" + suffix] = np.where(ok2, np.where(ctrl[nxt] == 1, 1.0, -1.0), 0.0)
    cols["ctrl_since@" + suffix], cols["ctrl_until@" + suffix] = s, u


def _segment(players, team, ball, pitch, radii, labels):
    """Features of one period (contiguous frames), coordinates already in the home-attack frame."""
    L, W = pitch
    T = len(ball)
    P = players.astype(np.float64)
    has = np.isfinite(P[..., 0]) & (team >= 0)
    P = np.where(has[..., None], P, np.nan)
    V = _vel(P)
    spd = np.linalg.norm(V, axis=-1)
    with np.errstate(invalid="ignore"):
        spd = np.where(spd > PLAYER_SPEED_CAP, np.nan, spd)
    cols = {}

    # ---- ball
    B = ballmod.clean(ball)
    present = np.isfinite(B[:, 0])
    vB = _vel(B)
    bspeed = np.nan_to_num(ballmod.speed(B), nan=0.0).clip(0, 30)
    d = np.linalg.norm(P - B[:, None, :], axis=-1)
    d = np.where(np.isfinite(d), d, np.inf)
    cols["ball_present"] = present * 1.0
    cols["ball_x"] = np.where(present, B[:, 0] / L - 0.5, 0.0)
    cols["ball_y"] = np.where(present, B[:, 1] / W - 0.5, 0.0)
    cols["ball_absy"] = np.abs(cols["ball_y"])
    cols["ball_speed"] = np.where(present, bspeed / 10.0, 0.0)
    out_m = np.max(np.stack([-B[:, 0], B[:, 0] - L, -B[:, 1], B[:, 1] - W], 1), axis=1)
    cols["ball_out_dist"] = np.where(present, np.clip(out_m, -5, 5) / 5.0, 0.0)
    d1 = {}
    for k, side in ((0, "H"), (1, "A")):
        dk = np.where(team == k, d, np.inf)
        order = np.argsort(dk, axis=1)
        srt = np.take_along_axis(dk, order, axis=1)
        a, b2 = np.minimum(srt[:, 0], 40.0), np.minimum(srt[:, 1], 40.0)
        d1[side] = a
        vp = V[np.arange(T), order[:, 0]]
        rel = np.linalg.norm(vB - vp, axis=1)
        rel = np.nan_to_num(rel, nan=15.0).clip(0, 15)
        cols["rel_" + side] = np.where(present, rel / 15.0, 1.0)
        cols["logd1_" + side] = np.where(present, np.log1p(a) / np.log(41.0), 1.0)
        cols["logd2_" + side] = np.where(present, np.log1p(b2) / np.log(41.0), 1.0)
        for r in (3, 5, 10):
            cols["n{}_{}".format(r, side)] = np.where(present, (dk <= r).sum(1) / 4.0, 0.0)
    cols["d_diff"] = np.where(present, np.clip(d1["A"] - d1["H"], -20, 20) / 10.0, 0.0)
    for r, label in zip(radii, labels):
        _control(cols, label, r, present, d1)
    s, last, ok = _since(present)
    u, nxt, ok2 = _until(present)
    cols["ball_since"], cols["ball_until"] = s, u
    bx = np.nan_to_num(B[:, 0] / L - 0.5)
    cols["ball_last_x"] = np.where(ok, bx[last], 0.0)
    cols["ball_next_x"] = np.where(ok2, bx[nxt], 0.0)
    cols["ball_last_out"] = np.where(ok, cols["ball_out_dist"][last], 0.0)
    cols["ball_next_out"] = np.where(ok2, cols["ball_out_dist"][nxt], 0.0)
    by = np.nan_to_num(np.abs(B[:, 1] / W - 0.5))
    cols["ball_last_absy"] = np.where(ok, by[last], 0.0)
    cols["ball_next_absy"] = np.where(ok2, by[nxt], 0.0)

    # ---- shape
    base = {}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, side in ((0, "h"), (1, "a")):
            mk = team == k
            x = np.where(mk, P[..., 0], np.nan) / L - 0.5
            y = np.where(mk, P[..., 1], np.nan) / W - 0.5
            sp = np.where(mk, spd, np.nan)
            vx = np.where(mk, V[..., 0], np.nan)
            n = np.isfinite(x).sum(1).astype(np.float64)
            st = {"n": n / 11.0, "cx": np.nanmean(x, 1), "cy": np.nanmean(y, 1),
                  "sx": np.nanstd(x, 1), "sy": np.nanstd(y, 1),
                  "x10": _nanpct(x, 10), "x90": _nanpct(x, 90),
                  "y10": _nanpct(y, 10), "y90": _nanpct(y, 90),
                  "speed": np.nanmean(sp, 1) / 3.0, "vx": np.nanmean(vx, 1) / 3.0,
                  "own_half": np.nanmean(np.where(np.isfinite(x), (x < 0) if k == 0 else (x > 0), np.nan), 1)}
            few = n < 4
            for key in st:
                if key != "n":
                    st[key][few] = np.nan
            st["len"] = st["x90"] - st["x10"]
            st["wid"] = st["y90"] - st["y10"]
            for key, val in st.items():
                base["{}_{}".format(side, key)] = val
        base["gap_cx"] = base["h_cx"] - base["a_cx"]
        base["front_back"] = base["h_x90"] - base["a_x10"]
        base["back_front"] = base["a_x90"] - base["h_x10"]
        base["speed_diff"] = base["h_speed"] - base["a_speed"]
        allsp = np.nanmean(spd, 1)
        centre = np.array([L / 2, W / 2])
        with np.errstate(invalid="ignore"):
            base["n_centre"] = (np.linalg.norm(P - centre, axis=-1) <= 9.15).sum(1) / 3.0
    for w in (25, 75):
        for key, val in base.items():
            cols["{}@{}".format(key, w)] = _rolling_mean(val[:, None], w)[:, 0]
    cols["pspeed1"] = _rolling_mean(allsp[:, None], FPS)[:, 0] / 3.0
    cols["pspeed@75"] = _rolling_mean(allsp[:, None], 75)[:, 0] / 3.0
    return cols, present


def build(players, team, ball, period, pitch, radii):
    """(X float32 [T, k], names, present [T]); NaN rows outside periods 1 and 2.

    radii: the model's three rungs, or ALL_RADII for the bank (names_bank)."""
    players = np.asarray(players, dtype=np.float64)
    ball = np.asarray(ball, dtype=np.float64)
    team = np.asarray(team)
    if team.ndim == 1:
        team = np.broadcast_to(team[None, :], players.shape[:2])
    right = home_attacks_right(players, team, period)
    L, W = pitch
    if tuple(radii) == ALL_RADII:
        names, labels = names_bank(), ["r{}".format(r) for r in radii]
    else:
        names, labels = names_rungs(), [str(i) for i in range(len(radii))]
    X = np.full((len(ball), len(names)), np.nan, dtype=np.float32)
    present = np.zeros(len(ball), dtype=bool)
    for per, home_right in right.items():
        rows = np.flatnonzero(period == per)
        a, b = rows[0], rows[-1] + 1
        Pp, Bp = players[a:b].copy(), ball[a:b].copy()
        if not home_right:
            Pp[..., 0], Pp[..., 1] = L - Pp[..., 0], W - Pp[..., 1]
            Bp[:, 0], Bp[:, 1] = L - Bp[:, 0], W - Bp[:, 1]
        cols, pres = _segment(Pp, team[a:b], Bp, pitch, radii, labels)
        X[a:b] = np.stack([np.asarray(cols[k], dtype=np.float64) for k in names], axis=1)
        present[a:b] = pres
    return X, names, present


# ------------------------------------------------------------------------------ C2, table 1.3

EVENT_COLUMNS = ("ev_still", "ev_since_still", "ev_kick_H", "ev_kick_A", "ev_since_kick")
KICK_WINDOW = 12


def series(players, team, ball, period):
    """Raw per-frame series of the C2 values and columns, over periods 1 and 2.

    (v float32 [T], dmin float32 [T], near_team int8 [T], p int8 [T]): ball speed, distance from the
    cleaned ball to the nearest player of either team and that player's team, ball seen.
    Not scaled, not capped: ball_speed and logd1 of the features are."""
    players = np.asarray(players, dtype=np.float64)
    ball = np.asarray(ball, dtype=np.float64)
    team = np.asarray(team)
    if team.ndim == 1:
        team = np.broadcast_to(team[None, :], players.shape[:2])
    T = len(ball)
    v = np.full(T, np.nan, dtype=np.float32)
    dmin = np.full(T, np.nan, dtype=np.float32)
    near = np.full(T, -1, dtype=np.int8)
    p = np.zeros(T, dtype=np.int8)
    for per in (1, 2):
        rows = np.flatnonzero(period == per)
        if not len(rows):
            continue
        a, b = rows[0], rows[-1] + 1
        B = ballmod.clean(ball[a:b])
        seen = np.isfinite(B[:, 0])
        Q = np.where((team[a:b] >= 0)[..., None], players[a:b], np.nan)
        d = np.linalg.norm(Q - B[:, None, :], axis=-1)
        d = np.where(np.isfinite(d), d, np.inf)
        j = d.argmin(1)
        dm = d[np.arange(b - a), j]
        ok = seen & np.isfinite(dm)
        v[a:b] = ballmod.speed(B)
        dmin[a:b] = np.where(ok, dm, np.nan)
        near[a:b] = np.where(ok, team[a:b][np.arange(b - a), j], -1)
        p[a:b] = seen
    return v, dmin, near, p


def _event_segment(v, dmin, near, p, val):
    T = len(v)
    seen = p > 0
    with np.errstate(invalid="ignore"):
        below = seen & (v < val["v_still"])
        fast = v >= val["v_kick"]
    last = np.maximum.accumulate(np.where(~below, np.arange(T), -1))
    run = np.arange(T) - last                    # length of the run of `below` ending at t
    still = below & (run >= val["T_still"])
    cum = np.r_[0, np.cumsum(fast)]
    stop = np.minimum(np.arange(T) + KICK_WINDOW, T)
    fast_ahead = cum[stop] > cum[np.arange(T)]
    prev = np.r_[False, still[:-1]]
    with np.errstate(invalid="ignore"):
        close = dmin <= val["d_player"]
    kick = prev & fast_ahead & close & seen
    out = {"ev_still": still * 1.0}
    out["ev_since_still"] = _since(still)[0]
    for k, name in ((0, "ev_kick_H"), (1, "ev_kick_A")):
        c = np.convolve((kick & (near == k)) * 1.0, np.ones(KICK_WINDOW))[:T]
        out[name] = ((c > 0) & seen) * 1.0
    out["ev_since_kick"] = _since(kick)[0]
    return np.stack([out[k] for k in EVENT_COLUMNS], axis=1)


def event_columns(v, dmin, near, p, period, val):
    """[T, 5] float32 columns of table 1.3 per period, NaN elsewhere.

    val: {"v_still", "T_still", "v_kick", "d_player"}."""
    v, dmin, near, p = (np.asarray(a) for a in (v, dmin, near, p))
    out = np.full((len(v), len(EVENT_COLUMNS)), np.nan, dtype=np.float32)
    for per in (1, 2):
        rows = np.flatnonzero(period == per)
        if len(rows):
            a, b = rows[0], rows[-1] + 1
            out[a:b] = _event_segment(v[a:b], dmin[a:b], near[a:b], p[a:b], val)
    return out
