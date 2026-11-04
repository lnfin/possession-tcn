"""Step M4: the whole path of a copy on the Mac, one thread, open matches only. Several short runs.

    python -m analysis.clean.smoke copies --root BANK
    python -m analysis.clean.smoke fallback --root BANK_FALLBACK
    python -m analysis.clean.smoke train --root BANK --out OUT --variant S0
    python -m analysis.clean.smoke decode --root BANK --out OUT
    python -m analysis.clean.smoke quantiles --root BANK --file analysis/clean/m4_quantiles.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from analysis.clean import bank, grid, metric
from mtah.external import reference
from mtah.seqmodel import net as seqnet
from mtah.transfer import features
from mtah.transfer_mt import decode, net

ROOT = Path(__file__).resolve().parents[2]
MATCHES = ("metrica1", "J03WMX", "J03WR9")
CENTRE = dict(channels=64, blocks=8, lr=2e-3, dropout=0.1, wd=1e-4, r0=0.75)


def copies(a):
    bank.write_refs(a.root, reference.RefParams(out_lag=3, cluster=39))
    jobs = bank.jobs_of("{0}:1:0,{0}:2:0,{0}:3:0".format("+".join(MATCHES)))
    bank.build(a.root, jobs, 1)
    print("gate G on these copies:", bank.gate(a.root, bank.jobs_of("{0}:1:0,{0}:2:0".format("+".join(MATCHES)))))


def fallback(a):
    bank.write_refs(a.root, reference.RefParams(out_lag=3, cluster=39))
    bank.build(a.root, bank.jobs_of("metrica1+J03WMX:1:0"), 1, True)
    for name in ("metrica1", "J03WMX"):
        base = bank.stem(a.root, name, 1, 0)
        X = np.load(str(base) + ".x.npy", mmap_mode="r")
        y = np.load(str(base) + ".y.npy", mmap_mode="r")[:, 0]
        seen = X[:, features.names_bank().index("ball_present")]
        print(name, "ball seen on D:", float(np.asarray(seen)[np.asarray(y) == 2].mean()))


def train(a):
    cmd = [sys.executable, str(ROOT / "training" / "train_clean.py"), "--root", a.root, "--out", a.out,
           "--hold", "metrica1", "--variant", a.variant, "--seed", "0", "--steps", "50", "--batch", "4",
           "--chunk", "1500", "--channels", "64", "--blocks", "8", "--lr", "2e-3", "--dropout", "0.1",
           "--weight-decay", "1e-4", "--r0", "0.75", "--g-seeds", "1", "--only", "J03WMX+J03WR9"]
    subprocess.run(cmd, check=True)


def decode_time(a):
    """Numpy forward against the torch OOF, then decoder and scorer times per copy."""
    w = seqnet.load(Path(a.out) / "w_C1_metrica1_s0.npz")
    base = str(bank.stem(a.root, "metrica1", 2, 0))
    period = np.load(base + ".per.npy")
    X = np.asarray(np.load(base + ".x.npy", mmap_mode="r")[:, features.bank_columns(0.75)], dtype=np.float64)
    t0 = time.time()
    P, aux = net.ours([w], X, features.names_rungs(), period)
    print("numpy forward of a copy: {:.1f} s".format(time.time() - t0))
    with np.load(Path(a.out) / "oof_C1_metrica1_s0.npz") as z:
        Pt, lg = z["metrica1_g2_s0/P"], z["metrica1_g2_s0/sw"]
    ok = period > 0
    print("max |numpy - torch| P:", float(np.abs(P[ok] - Pt[ok]).max()), " aux:", float(np.abs(aux[ok, 0] - lg[ok]).max()))
    y = np.load(base + ".y.npy")
    per = grid.periods_of(period)
    b = 1.0 / (1.0 + np.exp(-aux[:, 0]))
    t0 = time.time()
    lab = decode.decode_periods(P, period, 24.0, 36.0, 0.0, None, 0.0)
    t1 = time.time()
    lab2 = decode.decode_periods(P, period, 24.0, 36.0, 4.0, b, 2.0)
    t2 = time.time()
    r = [metric.spell_macro(reference.to_chars(y[:, e]), lab, per) for e in range(4)]
    t3 = time.time()
    print("decode const {:.2f} s, variable {:.2f} s, score 4 references {:.2f} s".format(t1 - t0, t2 - t1, t3 - t2))
    print("labels:", {k: int((lab == k).sum()) for k in "HADX"}, "spell, macro by reference:", np.round(r, 3).tolist())
    print("labels with kappa:", {k: int((lab2 == k).sum()) for k in "HADX"})


def quantiles(a):
    names = sorted(p.name[:-len(".x.npy")] for p in Path(a.root).glob("*.x.npy"))
    Z = []
    for n in names:
        per = np.load(Path(a.root) / (n + ".per.npy"))
        Z.append(np.array(np.load(Path(a.root) / (n + ".x.npy"), mmap_mode="r")[per > 0], dtype=np.float32))
    Z = np.concatenate(Z)
    lo, hi = [], []
    for k in range(Z.shape[1]):
        x = Z[:, k]
        x = x[np.isfinite(x)]
        lo.append(float(np.percentile(x, 0.1)))
        hi.append(float(np.percentile(x, 99.9)))
    Path(a.file).write_text(json.dumps({"names": features.names_bank(), "lo": lo, "hi": hi, "copies": names}))
    print("quantiles of", len(names), "copies,", Z.shape[1], "columns")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--variant", default="S0")
    ap.add_argument("--file", default="")
    a = ap.parse_args()
    {"copies": copies, "fallback": fallback, "train": train, "decode": decode_time, "quantiles": quantiles}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
