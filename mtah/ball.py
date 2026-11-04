"""Ball track cleaning: a position implying more than max_speed to both neighbours is dropped."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BallParams:
    max_speed: float              # m/s; a frame implying more than this to both neighbours is a spike


BALL = BallParams(max_speed=40.0)


def speed(ball_m, fps=25):
    """Per-frame ball speed in m/s from the previous frame; NaN where either frame is missing."""
    return np.r_[np.nan, np.linalg.norm(np.diff(ball_m, axis=0), axis=1) * fps]


def clean(ball_m, p=BALL, fps=25):
    """Copy of ball_m [N, 2] (metres) without single-frame spikes."""
    B = np.array(ball_m, dtype=np.float64)
    step = np.linalg.norm(np.diff(B, axis=0), axis=1) * fps
    v_in, v_out = np.r_[np.nan, step], np.r_[step, np.nan]
    with np.errstate(invalid="ignore"):
        spike = (v_in > p.max_speed) & (v_out > p.max_speed)
    B[spike] = np.nan
    return B
