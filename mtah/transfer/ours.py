"""The provided files in the layout of features.build.

The blocks only bound where we label. Each block's play starts at the earliest Kick Off (Start) tag
inside it; everything before that is D by Law 8 (L2), and the network starts there.
"""

import numpy as np

from .. import data, paths
from ..third_party import games


def blocks():
    """[(half, first, last)] of the graded video frames, inclusive."""
    return [tuple(b) for b in games.half_video_blocks(paths.GAME)]


def play_starts(tags, block_list):
    """{half: t0}: the earliest Kick Off (Start) tag of SET PIECES inside the block, else its first frame."""
    out = {}
    code = tags["code"].astype(str).str.upper() == "SET PIECES"
    kind = tags["Type"].astype(str).str.lower() == "kick off (start)"
    frames = tags["start_frame"][code & kind].to_numpy()
    for half, first, last in block_list:
        inside = frames[(frames >= first) & (frames <= last)]
        out[half] = int(inside.min()) if len(inside) else int(first)
    return out


def block_of(n, block_list):
    """[n] int8: the block's half on its frames, 0 elsewhere."""
    period = np.zeros(n, dtype=np.int8)
    for half, first, last in block_list:
        period[first:last + 1] = half
    return period


def play_period(block, t0, block_list):
    """block_of without the frames before t0: the period of the network."""
    period = block.copy()
    for half, first, _ in block_list:
        period[first:t0[half]] = 0
    return period


def match_arrays(block_list):
    """(players [N, S, 2] m, slot_team [S], ball [N, 2] m, pitch) of the match."""
    players, ball, slot_team = data.load_tracking()
    pitch = data.pitch_m()
    scale = np.array(pitch, dtype=np.float64)
    P = np.asarray(players[..., :2], dtype=np.float64) * scale
    B = np.asarray(ball, dtype=np.float64) * scale
    return P, np.asarray(slot_team, dtype=np.int8), B, pitch
