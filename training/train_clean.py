"""One script for S0, C1, C2, C7 and their combination.

    python training/train_clean.py --root BANK --out OUT --hold metrica1 --variant S0 --seed 0 --device cuda

--hold: a unit of section 0.4 (out-of-fold predictions on its T1, T2 copies), "all" (the final fit, no
predictions), "src:I" / "src:M" (train on one source, predict on the other, step P8).

Variants: S0 the base; C1 boundary head and frame weight; C2 the five columns of table 1.3; C7 targets
R(75). "+" combines. Hyperparameters are arguments, chosen on step P3b.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.clean import c2, metric  # noqa: E402
from mtah.seqmodel import net  # noqa: E402
from mtah.transfer import features  # noqa: E402
from mtah.transfer_mt import targets  # noqa: E402
from training.tcn import TCN, to_numpy  # noqa: E402

BOUND = 1.0                   # weight of the boundary term, C1
G_SEEDS = range(6)
EVAL_COPIES = ((2, 0), (2, 1), (3, 0), (3, 1))


def stem(root, name, g, s):
    return "{}/{}_g{}_s{}".format(root, name, g, s)


def split(hold):
    """(training matches by source {M, I}, held-out matches)."""
    if hold == "all":
        keep, ev = metric.GAMES_TRAIN, ()
    elif hold == "src:I":
        keep, ev = metric.IDSSE, metric.METRICA
    elif hold == "src:M":
        keep, ev = metric.METRICA, metric.GAMES_EVAL[3:]
    else:
        keep = tuple(n for n in metric.GAMES_TRAIN if n not in metric.UNIT_HOLD[hold])
        ev = metric.UNITS[hold]
    src = {"M": [n for n in keep if n in metric.METRICA], "I": [n for n in keep if n not in metric.METRICA]}
    return {k: v for k, v in src.items() if v}, ev


class Copy:
    """One bank copy: float16 features and labels are memmaps, shared between processes."""

    def __init__(self, root, name, g, s, cols, label_col):
        base = stem(root, name, g, s)
        self.base = base
        self.X = np.load(base + ".x.npy", mmap_mode="r")
        self.y = np.load(base + ".y.npy", mmap_mode="r")[:, label_col]
        self.period = np.load(base + ".per.npy")
        self.cols = cols
        self.segs = []
        for per in (1, 2):
            rows = np.flatnonzero(self.period == per)
            self.segs.append((int(rows[0]), int(rows[-1]) + 1))
        self.extra = None
        self.fb = None
        self.aux = None

    def add_c1(self):
        y = np.asarray(self.y)
        self.fb = np.ones(len(y), dtype=np.float32)
        self.aux = np.zeros((len(y), 2, 2), dtype=np.float32)
        for a, b in self.segs:
            yy = y[a:b]
            d = targets.distance_to(targets.changes(yy), 10 ** 4).astype(np.float64)
            self.fb[a:b] = 1.0 + BOUND * np.exp(-(d / targets.K_BOUND) ** 2)
            t = targets._segment_targets(yy)
            for j, k in enumerate(targets.AUX):
                self.aux[a:b, j, 0], self.aux[a:b, j, 1] = t[k]

    def add_c2(self, val):
        f = np.load(self.base + ".f.npy", mmap_mode="r")
        i = np.load(self.base + ".i.npy", mmap_mode="r")
        self.extra = features.event_columns(f[:, 0], f[:, 1], i[:, 0], i[:, 1], self.period, val)

    def chunk(self, s, n):
        x = np.array(self.X[s:s + n])[:, self.cols]
        return x if self.extra is None else (x, self.extra[s:s + n])


def col_stats(copies, cols):
    """Mean, std over period > 0, NaN skipped, from the per-copy sums of the bank (float16 values)."""
    n = s1 = s2 = 0.0
    for c in copies:
        z = np.load(c.base + ".st.npz")
        n, s1, s2 = n + z["n"][cols], s1 + z["s1"][cols], s2 + z["s2"][cols]
    if copies[0].extra is not None:
        ex_n = ex1 = ex2 = 0.0
        for c in copies:
            E = c.extra[c.period > 0].astype(np.float64)
            ex_n, ex1, ex2 = ex_n + len(E), ex1 + E.sum(0), ex2 + (E * E).sum(0)
        n, s1, s2 = np.r_[n, np.full(E.shape[1], ex_n)], np.r_[s1, ex1], np.r_[s2, ex2]
    mean = s1 / np.maximum(n, 1)
    std = np.maximum(np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0.0)), 1e-2)
    return mean, std


def to_device(batch, device, mean_t, std_t):
    """[B, k, T] standardised float32 on the device; NaN -> 0 after the standardisation."""
    if isinstance(batch[0], tuple):
        x = torch.from_numpy(np.stack([b[0] for b in batch])).to(device).float()
        ex = torch.from_numpy(np.stack([b[1] for b in batch])).to(device).float()
        x = torch.cat([x, ex], dim=2)
    else:
        x = torch.from_numpy(np.stack(batch)).to(device).float()
    x = (x - mean_t) / std_t
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).permute(0, 2, 1)


def train(by_source, a, mean, std, names, use_boundary, log):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    n_aux = len(targets.AUX) if use_boundary else 0
    model = TCN(len(names), a.channels, a.blocks, a.dropout, n_aux).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    rf = model.receptive()
    ce = nn.CrossEntropyLoss(reduction="none")
    mean_t = torch.tensor(mean, dtype=torch.float32, device=a.device)
    std_t = torch.tensor(std, dtype=torch.float32, device=a.device)
    sources = sorted(by_source)
    t0 = time.time()
    model.train()
    for step in range(a.steps):
        xb, yb, wb, fb, ab = [], [], [], [], []
        for _ in range(a.batch):
            matches = by_source[sources[rng.randint(len(sources))]]
            copies = list(matches.values())[rng.randint(len(matches))]
            c = copies[rng.randint(len(copies))]
            usable = [seg for seg in c.segs if seg[1] - seg[0] >= a.chunk]
            lens = np.array([z - s for s, z in usable], dtype=np.float64)
            p0, p1 = usable[rng.choice(len(usable), p=lens / lens.sum())]
            s = p0 + rng.randint(0, p1 - p0 - a.chunk + 1)
            y = np.array(c.y[s:s + a.chunk])
            w = (y >= 0).astype(np.float32)
            if s > p0:
                w[:rf] = 0
            if s + a.chunk < p1:
                w[-rf:] = 0
            xb.append(c.chunk(s, a.chunk))
            yb.append(np.maximum(y, 0).astype(np.int64))
            wb.append(w)
            if use_boundary:
                fb.append(c.fb[s:s + a.chunk])
                ab.append(c.aux[s:s + a.chunk])
        x = to_device(xb, a.device, mean_t, std_t)
        y = torch.from_numpy(np.stack(yb)).to(a.device)
        w = torch.from_numpy(np.stack(wb)).to(a.device)
        logits, aux = model(x)
        wf = w
        if use_boundary:
            wf = w * torch.from_numpy(np.stack(fb)).to(a.device)
        loss = (ce(logits, y) * wf).sum() / wf.sum().clamp(min=1.0)
        if use_boundary:
            A = torch.from_numpy(np.stack(ab)).to(a.device)          # [B, T, n_aux, 2]
            tgt, wt = A[..., 0].permute(0, 2, 1), (A[..., 1] * w[:, :, None]).permute(0, 2, 1)
            per = nn.functional.binary_cross_entropy_with_logits(aux, tgt, reduction="none")
            loss = loss + ((per * wt).sum(dim=(0, 2)) / wt.sum(dim=(0, 2)).clamp(min=1.0)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            log("  step {:5d} loss {:.4f} ({:.0f} s)".format(step, loss.item(), time.time() - t0))
    model.eval()
    return model


@torch.no_grad()
def predict(model, c, a, mean, std):
    """(P float32 [T, 3], boundary logit float32 [T] or None), NaN outside the periods."""
    mean_t = torch.tensor(mean, dtype=torch.float32, device=a.device)
    std_t = torch.tensor(std, dtype=torch.float32, device=a.device)
    T = len(c.period)
    P = np.full((T, 3), np.nan, dtype=np.float32)
    sw = np.full(T, np.nan, dtype=np.float32) if model.aux is not None else None
    for s, z in c.segs:
        x = to_device([c.chunk(s, z - s)], a.device, mean_t, std_t)
        logits, aux = model(x)
        P[s:z] = torch.softmax(logits, 1)[0].permute(1, 0).cpu().numpy()
        if sw is not None:
            sw[s:z] = aux[0, targets.AUX.index("sw")].cpu().numpy()
    return P, sw


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", required=True)
    ap.add_argument("--variant", default="S0")
    ap.add_argument("--tag", default="", help="extra name of the outputs: a configuration of step P3b")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--channels", type=int, required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--weight-decay", type=float, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--r0", type=float, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--g-seeds", type=int, default=len(G_SEEDS))
    ap.add_argument("--only", default="", help="smoke of step M4: these matches, the copies that exist")
    a = ap.parse_args()
    t0 = time.time()
    parts = a.variant.split("+")
    use_boundary, use_events, use_r75 = "C1" in parts, "C2" in parts, "C7" in parts
    label_col = metric.REFS.index("R75" if use_r75 else "R")
    cols = features.bank_columns(a.r0)
    names = features.names_rungs() + (list(features.EVENT_COLUMNS) if use_events else [])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = "{}{}_{}_s{}".format(a.variant, a.tag, a.hold.replace(":", ""), a.seed)
    logf = open(Path(a.out) / "log_{}.txt".format(tag), "w")

    def log(msg):
        print(msg, file=logf, flush=True)

    src, ev = split(a.hold)
    if a.only:
        keep = a.only.split("+")
        src = {k: [m for m in v if m in keep] for k, v in src.items()}
        src = {k: v for k, v in src.items() if v}
    val = None
    if use_events:
        val = c2.values(a.root, a.hold, range(a.g_seeds), a.only.split("+") if a.only else ())
        (Path(a.out) / "c2_{}.json".format(tag)).write_text(json.dumps(val))
    by_source = {}
    every = []
    for k, ms in src.items():
        by_source[k] = {}
        for m in ms:
            cs = [Copy(a.root, m, 1, s, cols, label_col) for s in range(a.g_seeds)]
            for c in cs:
                if use_boundary:
                    c.add_c1()
                if use_events:
                    c.add_c2(val)
            by_source[k][m] = cs
            every += cs
    mean, std = col_stats(every, cols)
    model = train(by_source, a, mean, std, names, use_boundary, log)
    w = to_numpy(model, mean, std, names)
    if use_boundary:
        w["aux_names"] = np.array(targets.AUX)
    net.save(Path(a.out) / "w_{}.npz".format(tag), w)
    oof = {}
    for name in ev:
        for g, s in EVAL_COPIES:
            if a.only and not Path(stem(a.root, name, g, s) + ".x.npy").is_file():
                continue
            c = Copy(a.root, name, g, s, cols, label_col)
            if use_events:
                c.add_c2(val)
            P, sw = predict(model, c, a, mean, std)
            key = "{}_g{}_s{}".format(name, g, s)
            oof[key + "/P"] = P
            if sw is not None:
                oof[key + "/sw"] = sw
    if oof:
        np.savez(Path(a.out) / "oof_{}.npz".format(tag), **oof)
    done = {"tag": tag, "seconds": time.time() - t0, "variant": a.variant, "hold": a.hold, "seed": a.seed,
            "config": {k: getattr(a, k) for k in ("steps", "channels", "blocks", "lr", "dropout", "weight_decay",
                                                  "batch", "chunk", "r0")}}
    with open(Path(a.out) / "times.jsonl", "a") as fh:
        fh.write(json.dumps(done) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
