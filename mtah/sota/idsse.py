"""IDSSE / DFL open data (Bassek et al., Sci Data 12, 195, 2025; CC BY 4.0): 7 Bundesliga matches
of 25 Hz positions for all players and the ball, with the provider's per-frame BallPossession
(1 home / 2 away) and BallStatus (1 alive / 0 dead) on the ball track.

Downloaded once from figshare into <cache>/external/idsse/ (never committed), parsed into one .npz
per match:

    players  float32 [T, 26 + 26, 2]  metres, centre origin, +x = the home team's attack; NaN = off
    team     int8    [52]             0 home, 1 away (slot k < 26 home)
    ball     float32 [T, 2]
    label    int8    [T]              0 home possession, 1 away, 2 dead
    half     int8    [T]              1 / 2
"""

import re
import subprocess
from pathlib import Path

import numpy as np

from .. import paths

MATCHES = {   # match id -> figshare file ids (meta, event, tracking), as kloppy.sportec lists them
    "J03WPY": (51643487, 51643505, 51643526),
    "J03WN1": (51643472, 51643496, 51643517),
    "J03WMX": (51643475, 51643493, 51643514),
    "J03WOH": (51643478, 51643499, 51643520),
    "J03WQQ": (51643484, 51643508, 51643529),
    "J03WOY": (51643481, 51643502, 51643523),
    "J03WR9": (51643490, 51643511, 51643532),
}
URL = "https://ndownloader.figshare.com/files/{}"
MAX_PER_TEAM = 26
PARSE_VERSION = 2   # 2: referees dropped

_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"')
_BALLFRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"[^>]*'
                        r'BallPossession="(\d)" BallStatus="(\d)"')
_TEAM = re.compile(r'<Team TeamId="([^"]+)"[^>]*Role="(\w+)"')


def folder():
    path = paths.cache_dir() / "external" / "idsse"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(match_id):
    out = {}
    for kind, file_id in zip(("meta", "event", "tracking"), MATCHES[match_id]):
        target = folder() / "{}_{}.xml".format(match_id, kind)
        if not target.is_file() or target.stat().st_size == 0:
            subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(target), URL.format(file_id)],
                           check=True)
        out[kind] = target
    return out


def _parse(files):
    roles = dict(_TEAM.findall(Path(files["meta"]).read_text(encoding="utf-8")))
    home = next(t for t, r in roles.items() if r == "home")
    tracks, ball = {}, {}
    cur, is_ball = None, False
    with open(files["tracking"], encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("<FrameSet"):
                section, team, person = _FRAMESET.search(line).groups()
                half = 1 if section == "firstHalf" else 2
                is_ball = team == "BALL"
                if is_ball:
                    cur = ball.setdefault(half, [])
                elif team not in roles:                  # the referees
                    cur = []
                else:
                    cur = tracks.setdefault((half, 0 if team == home else 1, person), [])
                continue
            if not line.startswith("<Frame "):
                continue
            if is_ball:
                m = _BALLFRAME.search(line)
                cur.append((int(m[1]), float(m[2]), float(m[3]), int(m[4]), int(m[5])))
            else:
                m = _FRAME.search(line)
                cur.append((int(m[1]), float(m[2]), float(m[3])))
    out = {k: [] for k in ("players", "ball", "label", "half")}
    persons = {0: sorted({p for (_, t, p) in tracks if t == 0}),
               1: sorted({p for (_, t, p) in tracks if t == 1})}
    for half in (1, 2):
        b = np.array(ball[half], dtype=np.float64)
        n0, T = int(b[0, 0]), len(b)
        if not np.array_equal(b[:, 0], np.arange(n0, n0 + T)):
            raise ValueError("ball frames not contiguous")
        P = np.full((T, 2 * MAX_PER_TEAM, 2), np.nan, dtype=np.float32)
        for (h, t, person), rows in tracks.items():
            if h != half:
                continue
            r = np.array(rows, dtype=np.float64)
            idx = r[:, 0].astype(np.int64) - n0
            ok = (idx >= 0) & (idx < T)
            P[idx[ok], t * MAX_PER_TEAM + persons[t].index(person)] = r[ok, 1:3]
        # +x = the home team's attack: the home outfield block sits at negative x at its own kick-off
        home_x = np.nanmean(P[:250, :MAX_PER_TEAM, 0])
        sign = 1.0 if home_x < 0 else -1.0
        P *= sign
        bxy = (b[:, 1:3] * sign).astype(np.float32)
        poss, status = b[:, 3].astype(np.int8), b[:, 4].astype(np.int8)
        if not np.isin(poss, (1, 2)).all():
            raise ValueError("BallPossession outside 1, 2")
        label = np.where(status == 0, 2, np.where(poss == 2, 1, 0)).astype(np.int8)
        out["players"].append(P)
        out["ball"].append(bxy)
        out["label"].append(label)
        out["half"].append(np.full(T, half, dtype=np.int8))
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["team"] = np.repeat(np.array([0, 1], dtype=np.int8), MAX_PER_TEAM)
    return res


def load(match_id):
    """Parsed arrays of one match (see the module docstring), cached as .npz."""
    target = folder() / "{}_v{}.npz".format(match_id, PARSE_VERSION)
    if not target.is_file():
        np.savez_compressed(target, **_parse(download(match_id)))
    with np.load(target) as z:
        return {k: z[k] for k in z.files}
