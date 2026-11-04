"""The decision steps of pod.py end to end on a synthetic work folder: no match, no training.

    MTAH_FLOW=<empty dir> pytest tests/test_flow.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.clean import grid, metric  # noqa: E402
from mtah.seqmodel import net as seqnet  # noqa: E402
from mtah.transfer import features  # noqa: E402

WORK = os.environ.get("MTAH_FLOW")
T = 3000


def labels(r):
    y = np.zeros(T, dtype=np.int8)
    t = 0
    while t < T:
        n = int(r.integers(40, 300))
        y[t:t + n] = r.integers(0, 3)
        t += n
    return y


def write_copy(root, name, g, s, r, n_cols):
    base = "{}/{}_g{}_s{}".format(root, name, g, s)
    y = labels(r)
    per = np.r_[np.ones(T // 2), 2 * np.ones(T - T // 2)].astype(np.int8)
    from mtah.external import reference
    Y = [y, reference.merge(y, ((1, 0, T // 2 - 1), (2, T // 2, T - 1)), 25),
         reference.merge(y, ((1, 0, T // 2 - 1), (2, T // 2, T - 1)), 75), y]
    np.save(base + ".y.npy", np.stack(Y, 1).astype(np.int8))
    np.save(base + ".per.npy", per)
    np.save(base + ".b.npy", np.where(r.uniform(size=T) < 0.3, -1, r.integers(0, 2, T)).astype(np.int8))
    np.save(base + ".x.npy", r.normal(0, 1, (T, n_cols)).astype(np.float16))
    np.save(base + ".f.npy", r.uniform(0, 5, (T, 2)).astype(np.float32))
    np.save(base + ".i.npy", np.stack([r.integers(0, 2, T), np.ones(T)], 1).astype(np.int8))
    Path(base + ".params.json").write_text(json.dumps({"info": {"kind": "T3a" if g == 4 else "x", "seconds": 1}}))
    return y


def prob(y, r, sharp):
    P = r.dirichlet((1, 1, 1), T) * 0.5
    P[np.arange(T), y] += 0.5 + sharp
    return (P / P.sum(1, keepdims=True)).astype(np.float32)


def weights(n_cols, names, aux):
    r = np.random.default_rng(0)
    w = {"mean": np.zeros(n_cols), "std": np.ones(n_cols), "names": names, "dilations": np.array([1, 2]),
         "in_w": r.normal(0, .1, (4, n_cols)).astype(np.float32), "in_b": np.zeros(4, np.float32),
         "out_w": r.normal(0, .1, (3, 4)).astype(np.float32), "out_b": np.zeros(3, np.float32)}
    for i in range(2):
        w["blk{}_w1".format(i)] = r.normal(0, .1, (4, 4, 3)).astype(np.float32)
        w["blk{}_b1".format(i)] = np.zeros(4, np.float32)
        w["blk{}_w2".format(i)] = r.normal(0, .1, (4, 4)).astype(np.float32)
        w["blk{}_b2".format(i)] = np.zeros(4, np.float32)
    if aux:
        w["aux_w"], w["aux_b"] = r.normal(0, .1, (2, 4)).astype(np.float32), np.zeros(2, np.float32)
    return w


def build(work):
    r = np.random.default_rng(1)
    bank = Path(work, "bank")
    runs = Path(work, "runs")
    bank.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    n_cols = len(features.names_bank())
    variants = {"S0": 0.1, "C1": 0.6, "C2": 0.1, "C7": 0.05, "C1+C2": 0.7}
    ys = {}
    for name in metric.GAMES_EVAL:
        for g, s in grid.COPIES + ((4, 0), (4, 1), (4, 2), (4, 3)):
            ys[(name, g, s)] = write_copy(bank, name, g, s, r, n_cols)
    for v, sharp in variants.items():
        names = features.names_rungs() + (list(features.EVENT_COLUMNS) if "C2" in v else [])
        for u, ms in metric.UNITS.items():
            for seed in (0, 1):
                z = {}
                for m in ms:
                    for g, s in grid.COPIES:
                        key = "{}_g{}_s{}".format(m, g, s)
                        z[key + "/P"] = prob(ys[(m, g, s)], r, sharp)
                        if "C1" in v:
                            z[key + "/sw"] = r.normal(0, 1, T).astype(np.float32)
                np.savez(runs / "oof_{}_{}_s{}.npz".format(v, u, seed), **z)
                seqnet.save(runs / "w_{}_{}_s{}.npz".format(v, u, seed), weights(len(names), names, "C1" in v))
            if "C2" in v:
                (runs / "c2_{}_{}_s0.json".format(v, u)).write_text(json.dumps(
                    {"v_still": 1.0, "T_still": 3, "v_kick": 2.0, "d_player": 2.0}))
        for src, ev in (("I", metric.METRICA), ("M", metric.GAMES_EVAL[3:])):
            z = {}
            for m in ev:
                for g, s in grid.COPIES:
                    key = "{}_g{}_s{}".format(m, g, s)
                    z[key + "/P"] = prob(ys[(m, g, s)], r, sharp)
                    if "C1" in v:
                        z[key + "/sw"] = r.normal(0, 1, T).astype(np.float32)
            np.savez(runs / "oof_{}_src{}_s0.npz".format(v, src), **z)
        for seed in (0, 1, 2):
            seqnet.save(runs / "w_{}_all_s{}.npz".format(v, seed), weights(len(names), names, "C1" in v))
    state = {"train": {"steps": 10, "channels": 4, "blocks": 2, "lr": 1e-3, "dropout": 0.0, "wd": 0.0, "batch": 2,
                       "chunk": 100, "r0": 0.75}, "out_lag": 3, "cluster": 39, "fallback": False,
             "B1": {"poss_per_min": [1, 2]}, "gate_bank": {"pass": True}}
    Path(work, "state.json").write_text(json.dumps(state))


def step(work, name, *extra):
    subprocess.run([sys.executable, "-m", "analysis.clean.pod", name, "--work", work, "--price", "0.5", "--spent", "1"]
                   + list(extra), cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.mark.skipif(not WORK, reason="MTAH_FLOW not set")
def test_flow():
    build(WORK)
    step(WORK, "p4eval")
    st = json.loads(Path(WORK, "state.json").read_text())
    assert st["eps_s"] >= 0.019 and st["S0"]["cell"][0] > 0
    st["G0"]["pass"] = True
    Path(WORK, "state.json").write_text(json.dumps(st))
    step(WORK, "p6")
    st = json.loads(Path(WORK, "state.json").read_text())
    assert "C1" in st["gate_report"] and "M_star" in st
    print("accepted", st["accepted"], {k: round(v["T"], 3) for k, v in st["gate_report"].items()})
    st["accepted"], st["M_star"] = ["C1", "C2"], None
    Path(WORK, "state.json").write_text(json.dumps(st))
    step(WORK, "p7")
    st = json.loads(Path(WORK, "state.json").read_text())
    assert st["M_star"] in ("C1", "C2", "C1+C2")
    print("M*", st["M_star"])
    st["M_star"] = "C1"
    Path(WORK, "state.json").write_text(json.dumps(st))
    step(WORK, "p8")
    rep = json.loads(Path(WORK, "p8.json").read_text())
    assert rep["M_star_out"] in ("C1", "S0") and "base_star_transfer" in rep and "T3" in rep
    st = json.loads(Path(WORK, "state.json").read_text())
    st["M_star"] = "C1"
    Path(WORK, "state.json").write_text(json.dumps(st))
    step(WORK, "p10bands")
    step(WORK, "registry")
    reg = json.loads(Path(WORK, "registry.json").read_text())
    assert reg["model_id"] == "C1" and reg["use_boundary"] and Path(WORK, "weights", "seed2.npz").is_file()
