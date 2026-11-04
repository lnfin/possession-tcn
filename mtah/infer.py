"""The whole path of run.py, numpy only.

predict works on arrays; run on the provided files. r0, the decoder and the values of C2 come only
from decode.json next to the weights.
"""

import json
import os
from pathlib import Path

import numpy as np

from . import data
from .seqmodel import net as seqnet
from .transfer import features, ours
from .transfer_mt import decode, net

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def load_model(weights_dir):
    """(weights of every seed, decode config)."""
    weights_dir = Path(weights_dir)
    cfg = json.loads((weights_dir / "decode.json").read_text())
    weights = [seqnet.load(p) for p in sorted(weights_dir.glob("seed*.npz"))]
    if not weights:
        raise FileNotFoundError("no seed*.npz in {}".format(weights_dir))
    return weights, cfg


def model_inputs(players, team, ball, period, pitch, cfg):
    """(X [T, k], names) of the model on this config."""
    r0 = cfg["r0"]
    X, names, _ = features.build(players, team, ball, period, pitch, tuple(r0 * k for k in features.RUNG))
    if cfg["use_events"]:
        v, dmin, near, p = features.series(players, team, ball, period)
        E = features.event_columns(v, dmin, near, p, period,
                                   {k: cfg[k] for k in ("v_still", "T_still", "v_kick", "d_player")})
        X = np.concatenate([X, E], axis=1)
        names = names + list(features.EVENT_COLUMNS)
    return X, names


def predict(players, ball, slot_team, period, t0, pitch, model):
    """Labels H/A/D [T], X outside the blocks. period: the block's half on its frames, t0 {half: frame}.
    slot_team [S], or [T, S] for a team per frame."""
    weights, cfg = model
    n_blocks = [(int(h), int(np.flatnonzero(period == h)[0]), int(np.flatnonzero(period == h)[-1]))
                for h in (1, 2) if (period == h).any()]
    play = ours.play_period(period, t0, n_blocks)
    slot_team = np.asarray(slot_team, dtype=np.int8)
    team = np.where(np.isfinite(players[..., 0]), slot_team if slot_team.ndim == 2 else slot_team[None, :],
                    -1).astype(np.int8)
    X, names = model_inputs(players, team, ball, play, pitch, cfg)
    P, aux = net.ours(weights, X, names, play)
    boundary = None
    if cfg["use_boundary"]:
        logit = aux[:, 0]
        boundary = 1.0 / (1.0 + np.exp(-logit))
    labels = decode.decode_periods(P, play, cfg["switch"], cfg["dead"], cfg["kappa"], boundary,
                                   cfg["switch_min"])
    labels[(period > 0) & (play == 0)] = "D"
    return labels


def run(data_dir, blocks, weights_dir):
    """Labels [N] of the match in data_dir; blocks [(half, first, last)]."""
    os.environ["MTAH_DATA_DIR"] = str(data_dir)
    P, slot_team, B, pitch = ours.match_arrays(blocks)
    period = ours.block_of(len(B), blocks)
    t0 = ours.play_starts(data.load_smart_tagging(), blocks)
    return predict(P, B, slot_team, period, t0, pitch, load_model(weights_dir))
