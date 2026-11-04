"""The ball's surroundings: the covariates of the gaps of G (schema 1.7, step 4)."""

import numpy as np

FPS = 25


def ball_state(players, has, ball, crowd_r_m):
    """(n_c [T] players within crowd_r_m of the ball, speed [T] m/s capped at 30)."""
    P = np.where(has[..., None], players, np.nan)
    B = np.asarray(ball, dtype=np.float64)
    d = np.linalg.norm(P - B[:, None, :], axis=-1)
    d = np.where(np.isfinite(d), d, np.inf)
    n_c = (d <= crowd_r_m).sum(1)
    vb = np.full_like(B, np.nan)          # backward: the last tracked frame before a gap has a speed
    vb[4:] = (B[4:] - B[:-4]) * (FPS / 4.0)
    speed = np.nan_to_num(np.linalg.norm(vb, axis=1), nan=0.0).clip(0, 30)
    return n_c, speed
