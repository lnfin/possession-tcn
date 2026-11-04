"""Numpy forward of the TCN with the optional auxiliary head: the trunk and main head of
mtah/seqmodel/net.py plus the 1x1 auxiliary head (aux_w, aux_b; channels in targets.AUX order). A
weight file without the auxiliary head gives aux = None."""

import numpy as np

from ..seqmodel import net
from ..seqmodel.net import _conv


def forward(w, X):
    """(log P [T, 3], aux logits [T, n_aux] or None) for one sequence X [T, k]."""
    x = (np.asarray(X, dtype=np.float64) - w["mean"]) / w["std"]
    x = np.where(np.isfinite(x), x, 0.0)
    h = x @ w["in_w"].T + w["in_b"]
    for i, dil in enumerate(w["dilations"]):
        z = np.maximum(_conv(h, w["blk{}_w1".format(i)], int(dil)) + w["blk{}_b1".format(i)], 0.0)
        h = h + z @ w["blk{}_w2".format(i)].T + w["blk{}_b2".format(i)]
    r = np.maximum(h, 0.0)
    logits = r @ w["out_w"].T + w["out_b"]
    logits = logits - logits.max(1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    aux = r @ w["aux_w"].T + w["aux_b"] if "aux_w" in w else None
    return logp, aux


def ours(weights, X, names, period):
    """(P [N, 3], aux logits [N, n_aux]) per period, averaged over an ensemble (aux: the logit mean)."""
    weights = weights if isinstance(weights, (list, tuple)) else [weights]
    if list(names) != weights[0]["names"]:
        raise ValueError("feature columns differ from the ones the model was trained on")
    n_aux = len(weights[0]["aux_b"]) if "aux_b" in weights[0] else 0
    P = np.full((len(X), 3), np.nan)
    aux = np.full((len(X), n_aux), np.nan)
    for per in (1, 2):
        rows = np.flatnonzero(period == per)
        if not len(rows):
            continue
        a, b = rows[0], rows[-1] + 1
        outs = [forward(w, X[a:b]) for w in weights]
        P[a:b] = np.mean([np.exp(lp) for lp, _ in outs], axis=0)
        if n_aux:
            aux[a:b] = np.mean([ax for _, ax in outs], axis=0)
    return P, aux


load = net.load
