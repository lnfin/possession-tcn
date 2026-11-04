"""IDSSE / DFL open data (Bassek et al., Sci Data 12, 195, 2025; CC BY 4.0): 7 matches of the
German Bundesliga 1/2 with 25 Hz optical tracking of every player and the ball, and the provider's
per-frame BallPossession (home / away) and BallStatus (alive / dead) channels.

Parsed by mtah/sota/idsse.py (download from figshare + XML parse); this module only converts that
to the layout of metrica_open.load(): normalised 0-1 coordinates, slots per team, periods, reference
codes, the ball height, and reads the events of R_def.
"""

import datetime as dt
import re

import numpy as np

from ..sota import idsse

MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
PITCH_M = (105.0, 68.0)    # PitchX / PitchY in all seven meta files
FPS = 25

_BALL_Z = re.compile(r'<Frame N="\d+"[^>]*? Z="([-\d.]+)"')
_SECTION = re.compile(r'<FrameSet GameSection="(\w+)"')
_TIME = re.compile(r'EventTime="([^"]+)"')
_FAILED = re.compile(r'\sEvaluation="unsuccessful"')


def folder():
    return idsse.folder()


def ball_z(match_id):
    """[T] ball height (m) in the frame order of load(); NaN where the frame has none."""
    path = folder() / "{}_ball_z.npy".format(match_id)
    if not path.is_file():
        z, cur = {}, None
        with open(folder() / "{}_tracking.xml".format(match_id), encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("<FrameSet"):
                    cur = z.setdefault(_SECTION.search(line)[1], []) if 'TeamId="BALL"' in line else None
                elif cur is not None and line.startswith("<Frame "):
                    m = _BALL_Z.search(line)
                    cur.append(float(m[1]) if m else np.nan)
        np.save(path, np.array(z["firstHalf"] + z["secondHalf"], dtype=np.float32))
    return np.load(path)


def load(match_id):
    """dict as metrica_open.load, plus "label" (provider codes 0 H / 1 A / 2 D, -1 outside), "ball_z"."""
    z = idsse.load(match_id)
    L, W = PITCH_M
    P = z["players"].astype(np.float64)                # metres, centre origin, +x = home attack
    used = np.isfinite(P[..., 0]).any(0)
    P = P[:, used]
    team = z["team"][used].astype(np.int8)
    players = np.stack([(P[..., 0] + L / 2) / L, (W / 2 - P[..., 1]) / W], -1).astype(np.float32)
    B = z["ball"].astype(np.float64)
    ball = np.stack([(B[:, 0] + L / 2) / L, (W / 2 - B[:, 1]) / W], -1).astype(np.float32)
    half = z["half"]
    periods = tuple((h, int(np.flatnonzero(half == h)[0]), int(np.flatnonzero(half == h)[-1])) for h in (1, 2))
    label = z["label"].astype(np.int8)
    bz = ball_z(match_id)
    if len(bz) != len(ball):
        raise ValueError("{}: {} ball heights for {} frames".format(match_id, len(bz), len(ball)))
    return {"name": "idsse_" + match_id, "players": players, "ball": ball, "slot_team": team,
            "periods": periods, "events": None, "label": label, "ball_z": bz,
            "pitch_m": PITCH_M, "fps": FPS}


def events(match_id):
    """[(half, seconds after that half's KickOff, team 0/1)] of ShotAtGoal and of Play with an
    Evaluation other than "unsuccessful", from the DFL event XML."""
    meta = (folder() / "{}_meta.xml".format(match_id)).read_text(encoding="utf-8")
    home = re.search(r'<Team TeamId="([^"]+)"[^>]*Role="home"', meta)[1]
    text = (folder() / "{}_event.xml".format(match_id)).read_text(encoding="utf-8")
    t0, out = {}, []
    for blk in re.split(r"(?=<Event )", text)[1:]:
        tm = dt.datetime.fromisoformat(_TIME.search(blk)[1]).timestamp()
        ko = re.search(r'<KickOff [^>]*GameSection="(\w+)"', blk)
        if ko and ko[1] in ("firstHalf", "secondHalf"):
            t0.setdefault(1 if ko[1] == "firstHalf" else 2, tm)
        m = re.search(r'<(Play|ShotAtGoal) [^>]*?Team="([^"]+)"[^>]*>', blk)
        if m and not _FAILED.search(m[0]):
            out.append((tm, 0 if m[2] == home else 1))
    half = [(2 if 2 in t0 and tm >= t0[2] else 1, tm, k) for tm, k in out]
    return [(h, tm - t0[h], k) for h, tm, k in half]
