"""The sequence model's forward pass in numpy float64 -- inference needs no torch.

Architecture (training/tcn.py builds the same in torch): a non-causal temporal CNN at 25 fps.

    x [T, k] standardised  -> 1x1 conv to C channels
    n blocks:  h = h + W2 . relu(dilated conv_3(h))      dilations 1, 2, 4, ...
    relu -> 1x1 conv to 3 logits (H, A, D) -> softmax

Each period is one sequence, zero-padded at its ends exactly as in training. Computed in float64
and rounded before decoding (decode.py), so last-bit differences between BLAS builds or machines do
not reach the labels.
"""

from pathlib import Path

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    w = {k: z[k] for k in z.files}
    w["names"] = [str(s) for s in w["names"]]
    return w


def save(path, w):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(w)
    out["names"] = np.array(w["names"])
    np.savez_compressed(path, **out)
    return path


def _conv(h, W, dilation):
    """Zero-padded 'same' dilated conv: out[t] = sum_k W[:, :, k] @ h[t + (k - K//2) * dilation]."""
    T = len(h)
    K = W.shape[2]
    out = np.zeros((T, W.shape[0]))
    for k in range(K):
        s = (k - K // 2) * dilation
        Wk = W[:, :, k].T
        if s >= 0:
            if s < T:
                out[:T - s] += h[s:] @ Wk
        elif -s < T:
            out[-s:] += h[:T + s] @ Wk
    return out


def forward(w, X):
    """Log-probabilities [T, 3] for one sequence X [T, k] (raw features, same columns as w['names'])."""
    x = (np.asarray(X, dtype=np.float64) - w["mean"]) / w["std"]
    x = np.where(np.isfinite(x), x, 0.0)
    h = x @ w["in_w"].T + w["in_b"]
    for i, dil in enumerate(w["dilations"]):
        z = np.maximum(_conv(h, w["blk{}_w1".format(i)], int(dil)) + w["blk{}_b1".format(i)], 0.0)
        h = h + z @ w["blk{}_w2".format(i)].T + w["blk{}_b2".format(i)]
    logits = np.maximum(h, 0.0) @ w["out_w"].T + w["out_b"]
    logits = logits - logits.max(1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(1, keepdims=True))
