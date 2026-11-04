"""The TCN in torch, training only. Its forward is mtah/transfer_mt/net.py in numpy."""

import numpy as np
import torch
from torch import nn


class TCN(nn.Module):
    def __init__(self, n_in, channels, n_blocks, dropout, n_aux):
        super().__init__()
        self.dilations = [2 ** i for i in range(n_blocks)]
        self.inp = nn.Conv1d(n_in, channels, 1)
        self.conv = nn.ModuleList(nn.Conv1d(channels, channels, 3, dilation=d, padding=d)
                                  for d in self.dilations)
        self.proj = nn.ModuleList(nn.Conv1d(channels, channels, 1) for _ in self.dilations)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Conv1d(channels, 3, 1)
        self.aux = nn.Conv1d(channels, n_aux, 1) if n_aux else None

    def forward(self, x):                       # x [B, k, T] -> logits [B, 3, T], aux [B, n_aux, T]
        h = self.inp(x)
        for conv, proj in zip(self.conv, self.proj):
            h = h + proj(self.drop(torch.relu(conv(h))))
        r = torch.relu(h)
        return self.out(r), (self.aux(r) if self.aux is not None else None)

    def receptive(self):
        return sum(self.dilations)               # frames on each side


def to_numpy(model, mean, std, names):
    g = lambda t: t.detach().cpu().numpy().astype(np.float32)   # noqa: E731
    w = {"mean": mean.astype(np.float64), "std": std.astype(np.float64), "names": names,
         "dilations": np.array(model.dilations, dtype=np.int64),
         "in_w": g(model.inp.weight)[:, :, 0], "in_b": g(model.inp.bias),
         "out_w": g(model.out.weight)[:, :, 0], "out_b": g(model.out.bias)}
    for i, (conv, proj) in enumerate(zip(model.conv, model.proj)):
        w["blk{}_w1".format(i)] = g(conv.weight)
        w["blk{}_b1".format(i)] = g(conv.bias)
        w["blk{}_w2".format(i)] = g(proj.weight)[:, :, 0]
        w["blk{}_b2".format(i)] = g(proj.bias)
    if model.aux is not None:
        w["aux_w"] = g(model.aux.weight)[:, :, 0]
        w["aux_b"] = g(model.aux.bias)
    return w
