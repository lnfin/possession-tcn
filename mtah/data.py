"""Loaders for the tactical assets.

Everything is indexed by VIDEO frame 0..N-1, i.e. by submission row. The ATD counts from 1, so its
record k lands in row k-1 here; nothing downstream has to remember that. N is the file's length.

    players   float32 [N, S, 3]   x, y (normalised 0-1, origin top-left), speed; NaN = not tracked
    ball      float32 [N, 2]      x, y; NaN = no ball
    slot_team int8    [S]         0 = home (H), 1 = away (A), from the XML
"""

import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import paths


# --------------------------------------------------------------------------------------- tracking

def _meta():
    return ET.parse(paths.require(paths.asset("atd_meta"))).getroot()


def slot_teams():
    """Team index (0 home, 1 away) of each ATD slot, in the channel order the raw file uses."""
    root = _meta()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team_of_player = {p.get("id"): team_ids.index(p.get("teamId")) for p in root.iter("Player")}
    # Channel order of the raw file: player<id>_x, in the order the spec lists them.
    order = [ref.get("playerChannelId")[len("player"):-len("_x")]
             for ref in root.iter("PlayerChannelRef")
             if ref.get("playerChannelId").endswith("_x")]
    return np.array([team_of_player[pid] for pid in order], dtype=np.int8)


def pitch_m():
    """(length, width) in metres from the XML's FieldSize (EPTS: Width is the long side)."""
    size = next(_meta().iter("FieldSize"))
    return float(size.findtext("Width")), float(size.findtext("Height"))


def home_attacks_right():
    """{half: True if the home team attacks towards +x}, from the XML's first team."""
    home = next(_meta().iter("Team"))
    value = None
    for par in home.iter("ProviderParameter"):
        if par.findtext("Name") == "attack_direction_first_half":
            value = par.findtext("Value")
    if value not in ("left_to_right", "right_to_left"):
        raise ValueError("no attack_direction_first_half for the home team in the ATD metadata")
    return {1: value == "left_to_right", 2: value != "left_to_right"}


def team_tags():
    """{team name, lower case: 0 home / 1 away}; Smart Tagging names the teams the same way."""
    return {t.findtext("Name").strip().lower(): k for k, t in enumerate(_meta().iter("Team"))}


def load_tracking():
    """(players [N,S,3], ball [N,2], slot_team [S]); N and S come from the files."""
    slot_team = slot_teams()
    n_slots = len(slot_team)
    raw = paths.require(paths.asset("atd_raw")).read_text()
    # "<id>:<x,y,s>;...;<x,y,s>:<bx,by,bz>"  ->  one flat CSV row of 1 + S*3 + 3 numbers
    table = pd.read_csv(io.StringIO(raw.replace(":", ",").replace(";", ",")),
                        header=None, dtype=np.float64, engine="c")
    n_cols = 1 + n_slots * 3 + 3
    if table.shape[1] != n_cols:
        raise ValueError("ATD rows have {} fields, expected {}".format(table.shape[1], n_cols))
    values = table.to_numpy()
    frame_id = values[:, 0].astype(np.int64)
    if not np.array_equal(frame_id, np.arange(1, len(values) + 1)):
        raise ValueError("ATD frame ids are not exactly 1..{}".format(len(values)))
    players = values[:, 1:1 + n_slots * 3].reshape(-1, n_slots, 3).astype(np.float32)
    ball = values[:, 1 + n_slots * 3:1 + n_slots * 3 + 2].astype(np.float32)
    return players, ball, slot_team


# ---------------------------------------------------------------------------------- smart tagging

def load_smart_tagging():
    """DataFrame: code, start_frame, end_frame (exclusive), start_s, end_s, Team, Half, Type, ..."""
    rows = []
    root = ET.parse(paths.require(paths.asset("smart_tagging"))).getroot()
    for inst in root.iter("instance"):
        start_s, end_s = float(inst.findtext("start")), float(inst.findtext("end"))
        row = {"code": inst.findtext("code").strip().upper(),
               "start_s": start_s, "end_s": end_s,
               "start_frame": int(round(start_s * paths.FPS)),
               "end_frame": int(round(end_s * paths.FPS))}
        for label in inst.findall("label"):
            row[label.findtext("group")] = label.findtext("text")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("start_frame", kind="stable").reset_index(drop=True)
