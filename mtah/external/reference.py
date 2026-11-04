"""H/A/D reference labels of the open matches: Metrica from events, by table 1.8 of the protocol.

Possession (Metrica's event definitions, documentation/events-definitions.pdf): the team of the
latest "owning" event -- SET PIECE, PASS, CARRY, RECOVERY, BALL LOST, BALL OUT, SHOT, FAULT
RECEIVED; CHALLENGE, CARD and SUBSTITUTION own nothing. A change of team therefore lands on the
new team's first touch (RECOVERY.start), and a touch that does not win the ball changes nothing.

Dead ball: every restart -- SET PIECE, or a goal kick (Metrica logs it as PASS / BALL LOST / SHOT
with subtype GOAL KICK) -- closes a dead stretch that opens at the stoppage, read off the latest
event before the restart, first matching row wins:

    SHOT, BALL OUT with GOAL                            end
    BALL OUT                                            max(start, end - out_lag)
    PASS, SHOT, CARRY, BALL LOST with OUT or ending off the pitch      end
    FAULT RECEIVED, CHALLENGE with FAULT, BALL LOST with FORCED / OFFSIDE / HAND BALL   max(start, end)
    anything else                                       latest end in the cluster

Of the events within `cluster` frames before the latest one, the strongest reason wins (goal > out
> referee), then the later stop. Each period is dead before its first event and after its last.

Versions of the reference (section 0.4): merge(y, L) is R(L); keep_runs is the mechanism of R_def.
In both a team run right after D stays, so the D frames are the same in every version.
"""

from dataclasses import dataclass

import numpy as np

H, A, D = 0, 1, 2
POSSESSION = {"SET PIECE", "PASS", "CARRY", "RECOVERY", "BALL LOST", "BALL OUT", "SHOT", "FAULT RECEIVED"}
NEUTRAL = {"CARD", "SUBSTITUTION", "CHALLENGE"}   # never the stoppage when looking back from a restart
REFEREE = {"FORCED", "OFFSIDE", "HAND BALL"}
MAX_LAG = 125
FPS = 25


@dataclass(frozen=True)
class RefParams:
    out_lag: int               # BALL OUT.end is this many frames after the crossing
    cluster: int               # events this close before the latest one are one moment


KINDS = ("referee", "out", "goal")
RANK = {k: i for i, k in enumerate(KINDS)}


def _off_pitch(x, y):
    return bool(np.isfinite(x) and np.isfinite(y) and (x < 0 or x > 1 or y < 0 or y > 1))


def is_restart(e):
    return e.type == "SET PIECE" or (e.type in ("PASS", "BALL LOST", "SHOT") and "GOAL KICK" in e.subtypes)


def stoppage_start(e, p):
    """(frame the ball is dead from, kind) for the latest event before a restart."""
    if "GOAL" in e.subtypes and e.type in ("SHOT", "BALL OUT"):
        return e.end, "goal"
    if e.type == "BALL OUT":
        return max(e.start, e.end - p.out_lag), "out"
    if e.type in ("PASS", "SHOT", "CARRY", "BALL LOST") and ("OUT" in e.subtypes or _off_pitch(e.ex, e.ey)):
        return e.end, "out"
    if e.type == "FAULT RECEIVED" or (e.type == "CHALLENGE" and "FAULT" in e.subtypes) or \
            (e.type == "BALL LOST" and e.subtypes & REFEREE):
        return max(e.start, e.end), "referee"
    return max(e.start, e.end), "unexplained"


def labels(game, p, return_log=False):
    """int8 [N]: 0 H, 1 A, 2 D inside the periods, -1 (X) outside; optionally the stoppage log."""
    n = len(game["ball"])
    out = np.full(n, -1, dtype=np.int8)
    log = []
    ev = game["events"].sort_values(["start", "end"], kind="stable")
    for period, first, last in game["periods"]:
        E = ev[(ev.period == period)].reset_index(drop=True)
        seg = np.full(last - first + 1, D, dtype=np.int8)
        own = E[E.type.isin(POSSESSION)]
        starts = np.clip(own.start.to_numpy() - first, 0, len(seg))
        teams = own.team.to_numpy()
        for k in range(len(own)):
            b = starts[k + 1] if k + 1 < len(own) else len(seg)
            seg[starts[k]:b] = teams[k]
        dead = np.zeros(len(seg), dtype=bool)
        dead[:max(0, int(E.start.iloc[0]) - first)] = True            # before the kick-off
        end_play = int(max(E.start.max(), E.end.max())) - first + 1
        dead[end_play:] = True                                          # after the final event
        for i in range(1, len(E)):
            r = E.iloc[i]
            if not is_restart(r):
                continue
            # the events just before the restart (Metrica often logs a RECOVERY by the taker on
            # the restart frame itself, and BALL OUT / BALL LOST of one moment in either order)
            before = E.iloc[:i]
            before = before[(before.start < r.start) &
                            ~(before.type.isin(NEUTRAL) & ~before.subtypes.map(lambda s: "FAULT" in s))]
            if not len(before):
                continue
            cluster = before[before.start >= before.start.max() - p.cluster]
            found = [(stoppage_start(e, p), e.type) for e in cluster.itertuples()]
            stops = [(RANK[k], s, t) for (s, k), t in found if k != "unexplained"]
            if stops:
                _, s, prev_type = max(stops)
                kind = KINDS[max(stops)[0]]
            else:
                s, prev_type, kind = int(cluster.end.max()), cluster.type.iloc[-1], "unexplained"
            s = min(s, int(r.start))
            if s < r.start:
                dead[s - first:int(r.start) - first] = True
            log.append({"period": period, "restart": int(r.start), "restart_type": r.type,
                        "restart_sub": "-".join(sorted(r.subtypes)), "stop": int(s), "kind": kind,
                        "prev_type": prev_type})
        seg[dead] = D
        out[first:last + 1] = seg
    return (out, log) if return_log else out


def merge(y, periods, length):
    """R(length): a team run shorter than length joins the previous team label, left to right."""
    def keep(a, z, k):
        return z - a >= length
    return keep_runs(y, periods, keep)


def keep_runs(y, periods, keep):
    """Team runs of y for which keep(a, z, team) is False join the previous team label; a run right
    after D, and D itself, stay. Frames (a, z) are absolute, z exclusive."""
    o = y.copy()
    for _, a0, b0 in periods:
        c = o[a0:b0 + 1]
        b = np.r_[0, np.flatnonzero(np.diff(c) != 0) + 1, len(c)]
        for a, z in zip(b[:-1], b[1:]):
            if c[a] != D and a > 0 and c[a - 1] != D and not keep(a + a0, z + a0, int(c[a])):
                c[a:z] = c[a - 1]
    return o


def metrica_def(game, y):
    """R_def of a Metrica match: a run stays only with a PASS or a SHOT of its team inside it."""
    E = game["events"]
    E = E[E.type.isin(["PASS", "SHOT"])]
    st, tm = E.start.to_numpy(), E.team.to_numpy()
    return keep_runs(y, game["periods"], lambda a, z, k: bool(((st >= a) & (st < z) & (tm == k)).any()))


def lag_of(ev, y, periods, max_lag=MAX_LAG):
    """The lag maximising the share of events inside a flag run of their own team; ties go to the
    smaller |lag|, then the smaller lag. Returns lag, share, event frames, event teams."""
    first = {h: a for h, a, _ in periods}
    last = {h: b for h, _, b in periods}
    fr = np.array([first[h] + int(round(t * FPS)) for h, t, _ in ev])
    tm = np.array([k for _, _, k in ev])
    hi = np.array([last[h] for h, _, _ in ev])
    best = (-1.0, 0)
    for lag in sorted(range(-max_lag, max_lag + 1), key=abs):
        f = np.clip(fr + lag, 0, hi)
        share = float((y[f] == tm).mean())
        if share > best[0]:
            best = (share, lag)
    return best[1], best[0], fr + best[1], tm


def idsse_def(ev, y, periods):
    """(R_def, lag, share) of an IDSSE match; ev from idsse_open.events."""
    lag, share, fr, tm = lag_of(ev, y, periods)
    return keep_runs(y, periods, lambda a, z, k: bool(((fr >= a) & (fr < z) & (tm == k)).any())), lag, share


def to_chars(codes):
    return np.array(["H", "A", "D", "X"])[np.where(codes < 0, 3, codes)]
