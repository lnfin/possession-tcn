"""Metrica Sports open sample data: three anonymised matches with synchronised tracking and events.

    https://github.com/metrica-sports/sample-data   (no licence file; the README asks to be
    responsible with the data and to acknowledge the source when used publicly)

    Sample Game 1, 2   CSV tracking (home and away files) + CSV events
    Sample Game 3      FIFA EPTS tracking (txt + XML metadata, read with kloppy) + JSON events

Everything is turned into one shape (row i = tracking frame i + 1):

    players    float32 [N, S, 2]  x, y normalised 0-1, origin top-left; NaN = not on
    ball       float32 [N, 2]     NaN = no ball
    slot_team  int8    [S]        0 = home, 1 = away
    keeper     int     [2]        home / away slot of the starting goalkeeper
    periods    ((1, first, last), (2, first, last)), inclusive rows
    events     DataFrame: team (0/1), type, subtypes (set of upper-case tokens), period, start, end
               (rows), sx, sy, ex, ey (normalised; NaN when not given)
    home_attacks_right  {1: bool, 2: bool}

The parse is cached as .npz/.pkl under $MTAH_DATA_DIR/cache/external/metrica/ (the data is
public, the cache is still never committed).
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths

GAMES = ("metrica1", "metrica2", "metrica3")
PITCH_M = (105.0, 68.0)    # README: "The dimensions of the field are the same ...: 105x68 meters"
FPS = 25


def external_dir():
    return Path(os.environ.get("MTAH_EXTERNAL_DIR") or paths.data_dir() / "external" / "metrica-sample-data")


def cache_root():
    """External parses."""
    return Path(os.environ.get("MTAH_EXTERNAL_CACHE") or paths.data_dir() / "cache" / "external")


def _cache(name):
    path = cache_root() / "metrica"
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def _tokens(sub):
    if sub is None or (isinstance(sub, float) and np.isnan(sub)):
        return frozenset()
    if isinstance(sub, str):
        return frozenset(s.strip().upper() for s in sub.split("-") if s.strip())
    if isinstance(sub, dict):
        sub = [sub]
    return frozenset(s["name"].strip().upper() for s in sub)


# ------------------------------------------------------------------------------- Sample Game 1, 2

def _csv_tracking(path):
    head = pd.read_csv(path, skiprows=2, nrows=0).columns
    names, k = [], 3
    cols = list(head)
    while k < len(cols):
        names.append(cols[k])
        k += 2
    table = pd.read_csv(path, skiprows=3, header=None)
    xy = table.iloc[:, 3:3 + 2 * len(names)].to_numpy(np.float32).reshape(len(table), len(names), 2)
    return table.iloc[:, 0].to_numpy(), table.iloc[:, 1].to_numpy(), names, xy


def _load_csv_game(g):
    d = external_dir() / "Sample_Game_{}".format(g)
    per, frame, hn, hxy = _csv_tracking(d / "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(g))
    per2, frame2, an, axy = _csv_tracking(d / "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(g))
    if not (np.array_equal(frame, frame2) and np.array_equal(frame, np.arange(1, len(frame) + 1))):
        raise ValueError("game {}: home and away tracking frames differ".format(g))
    # the last pair of each file is the ball
    assert hn[-1] == "Ball" and an[-1] == "Ball"
    ball = hxy[:, -1]
    players = np.concatenate([hxy[:, :-1], axy[:, :-1]], axis=1)
    slot_team = np.r_[np.zeros(len(hn) - 1), np.ones(len(an) - 1)].astype(np.int8)
    # starting goalkeeper: the first listed player of each team (Player11 / Player25 are the keepers)
    keeper = (0, len(hn) - 1)
    periods = tuple((p, int(np.flatnonzero(per == p)[0]), int(np.flatnonzero(per == p)[-1])) for p in (1, 2))

    ev = pd.read_csv(d / "Sample_Game_{}_RawEventsData.csv".format(g))
    events = pd.DataFrame({
        "team": (ev["Team"].str.strip().str.lower() == "away").astype(np.int8),
        "type": ev["Type"].str.strip().str.upper(),
        "subtypes": [_tokens(s) for s in ev["Subtype"]],
        "period": ev["Period"].astype(int),
        "start": ev["Start Frame"].astype(int) - 1,
        "end": ev["End Frame"].astype(int) - 1,
        "sx": ev["Start X"].astype(float), "sy": ev["Start Y"].astype(float),
        "ex": ev["End X"].astype(float), "ey": ev["End Y"].astype(float),
    })
    return players, ball, slot_team, keeper, periods, events


# ---------------------------------------------------------------------------------- Sample Game 3

def _load_game3():
    from kloppy import metrica
    d = external_dir() / "Sample_Game_3"
    ds = metrica.load_tracking_epts(meta_data=str(d / "Sample_Game_3_metadata.xml"),
                                    raw_data=str(d / "Sample_Game_3_tracking.txt"),
                                    coordinates="metrica")
    df = ds.to_df(engine="pandas")
    if not np.array_equal(df["frame_id"].to_numpy(), np.arange(1, len(df) + 1)):
        raise ValueError("game 3: tracking frames are not 1..N")
    home, away = ds.metadata.teams
    cols, team, keeper = [], [], []
    for k, t in enumerate((home, away)):
        for pl in t.players:
            if "{}_x".format(pl.player_id) not in df:
                continue
            if str(pl.starting_position).find("Goalkeeper") >= 0 and len(keeper) == k:
                keeper.append(len(cols))
            cols.append(pl.player_id)
            team.append(k)
    players = np.stack([df[["{}_x".format(c), "{}_y".format(c)]].to_numpy(np.float32) for c in cols], 1)
    ball = df[["ball_x", "ball_y"]].to_numpy(np.float32)
    per = df["period_id"].to_numpy()
    periods = tuple((p, int(np.flatnonzero(per == p)[0]), int(np.flatnonzero(per == p)[-1])) for p in (1, 2))

    raw = json.loads((d / "Sample_Game_3_events.json").read_text())["data"]
    names = {home.name: 0, away.name: 1}
    rows = []
    for e in raw:
        rows.append({"team": names[e["team"]["name"]], "type": e["type"]["name"].strip().upper(),
                     "subtypes": _tokens(e["subtypes"]), "period": int(e["period"]),
                     "start": int(e["start"]["frame"]) - 1, "end": int(e["end"]["frame"]) - 1,
                     "sx": np.nan if e["start"]["x"] is None else float(e["start"]["x"]),
                     "sy": np.nan if e["start"]["y"] is None else float(e["start"]["y"]),
                     "ex": np.nan if e["end"]["x"] is None else float(e["end"]["x"]),
                     "ey": np.nan if e["end"]["y"] is None else float(e["end"]["y"])})
    events = pd.DataFrame(rows)
    events["team"] = events["team"].astype(np.int8)
    return players, ball, np.array(team, dtype=np.int8), tuple(keeper), periods, events


# ----------------------------------------------------------------------------------------- public

def _attack_direction(players, slot_team, periods):
    """{period: home attacks +x}: the home team starts each period in its own (left/right) half."""
    out = {}
    for p, a, _ in periods:
        x = players[a:a + 5 * FPS, :, 0]
        with np.errstate(invalid="ignore"):
            out[p] = bool(np.nanmean(x[:, slot_team == 0]) < np.nanmean(x[:, slot_team == 1]))
    return out


def load(game):
    """dict with players, ball, slot_team, keeper, periods, events, home_attacks_right, pitch_m."""
    if game not in GAMES:
        raise ValueError("unknown game {!r}, expected one of {}".format(game, GAMES))
    npz, pkl = _cache("{}.npz".format(game)), _cache("{}_meta.pkl".format(game))
    if not (npz.is_file() and pkl.is_file()):
        g = int(game[-1])
        players, ball, slot_team, keeper, periods, events = _load_game3() if g == 3 else _load_csv_game(g)
        np.savez(npz, players=players, ball=ball, slot_team=slot_team)
        pkl.write_bytes(pickle.dumps({"keeper": keeper, "periods": periods, "events": events}))
    arrays = np.load(npz)
    meta = pickle.loads(pkl.read_bytes())
    out = {"name": game, "players": arrays["players"], "ball": arrays["ball"],
           "slot_team": arrays["slot_team"], "pitch_m": PITCH_M, "fps": FPS, **meta}
    out["home_attacks_right"] = _attack_direction(out["players"], out["slot_team"], out["periods"])
    return out
