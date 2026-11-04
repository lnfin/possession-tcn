"""One open match in the layout of the fallback generator (transfer/augment.py)."""

from dataclasses import dataclass

import numpy as np

from ..transfer_unified import simulate2


@dataclass(frozen=True)
class Match:
    players: np.ndarray          # [T, S, 2] metres
    ball: np.ndarray             # [T, 2] metres
    slot_team: np.ndarray        # [S] 0 home, 1 away
    labels: np.ndarray           # [T] "H", "A", "D", "X"
    period: np.ndarray           # [T] 1, 2, 0 outside
    pitch: tuple


def game_to_match(game, ref_codes, log, dead_params, rng_g):
    """Match of a game; its ball carries the dead ball of clean_ball where the clean ball vanishes."""
    L, W = game["pitch_m"]
    players = np.asarray(game["players"], dtype=np.float64) * np.array([L, W])
    ball = simulate2.clean_ball(game, ref_codes, log, dead_params, rng_g)
    labels = np.array(["H", "A", "D", "X"])[np.where(ref_codes < 0, 3, ref_codes)]
    return Match(players=players, ball=ball, slot_team=np.asarray(game["slot_team"], dtype=np.int8),
                 labels=labels, period=simulate2.period_of(game), pitch=(L, W))
