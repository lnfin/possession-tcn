"""The bank of copies: loader -> reference -> generator -> features, no substitution of paths.

    python -m analysis.clean.bank refs --root DIR                    # references of the 10 matches
    python -m analysis.clean.bank build --root DIR --spec g,s;... --workers N
    python -m analysis.clean.bank gate --root DIR --spec ...         # gate G of section 0.4

Copy seed c = 10**6 g + 10**3 m + s; team swap and mirror from default_rng([c, 2]): the swap before the
generator on game and labels, the mirror after it. Each copy is stored as

    <name>_g<g>_s<s>.x.npy      float16 [T, 137] features, NaN outside the periods
    <name>_g<g>_s<s>.y.npy      int8 [T, 4] references R, R25, R75, Rdef after the swap (2 = D, -1 = X)
    <name>_g<g>_s<s>.b.npy      int8 [T] the baseline's nearest-player codes, unsmoothed
    <name>_g<g>_s<s>.f.npy      float32 [T, 2] v, dmin
    <name>_g<g>_s<s>.i.npy      int8 [T, 2] near_team, p
    <name>_g<g>_s<s>.per.npy    int8 [T] period
    <name>_g<g>_s<s>.st.npz     per-column count, sum, sum of squares over period > 0
    <name>_g<g>_s<s>.params.json

The bank is deterministic in the seeds and is not copied off the pod.
"""

import argparse
import dataclasses
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from analysis.clean import baseline, metric, priors
from mtah.external import corrupt, reference
from mtah.transfer import augment, features, match
from mtah.transfer_unified import simulate2

CACHE_VERSION = 2
MAX_COPY_SECONDS = 300
GEN_KIND = {1: "G", 2: "T1", 3: "T2"}
GEN_KIND_FALLBACK = {1: "AUG", 2: "AUG1", 3: "T2"}
_GAMES = {}


def seed_of(g, m, s):
    return 10 ** 6 * g + 10 ** 3 * m + s


def stem(root, name, g, s):
    return Path(root) / "{}_g{}_s{}".format(name, g, s)


def swap_game(game, Y, swap):
    """(game, {references}) with home and away exchanged when swap."""
    if not swap:
        return game, Y
    game = dict(game, slot_team=(1 - np.asarray(game["slot_team"])).astype(np.int8))
    return game, {k: np.where(v >= 2, v, np.where(v < 0, v, 1 - v)).astype(np.int8) for k, v in Y.items()}


def kind_of(g, s, fallback):
    if g == 4:
        return "T3a" if s < 2 else "T3b"
    return (GEN_KIND_FALLBACK if fallback else GEN_KIND)[g]


def _params_dict(p):
    return dataclasses.asdict(p)


def make_copy(name, g, s, game, Y, log, fallback=False):
    """(players m [T, S, 2], team [T, S] or [S], ball m [T, 2], {references}, period, info, params)."""
    m = metric.GAMES_TRAIN.index(name)
    c = seed_of(g, m, s)
    kind = kind_of(g, s, fallback)
    p = priors.draw(kind, c)
    r = np.random.default_rng([c, 2])
    swap, mirror = r.random() < 0.5, r.random() < 0.5
    game, Y = swap_game(game, Y, swap)
    L, W = game["pitch_m"]
    period = simulate2.period_of(game)
    scale = np.array([L, W])
    if kind in ("G", "T1", "T3a"):
        players, ball, slot_team, info = simulate2.to_atd(game, Y["R"], log, p)
    elif kind in ("T2", "T3b"):
        players, ball, slot_team, info = corrupt.to_atd(game, Y["R"], log, p)
    if kind in ("AUG", "AUG1"):
        mt = match.game_to_match(game, Y["R"], log, p.dead, np.random.default_rng(c))
        Pm, team, Bm, _ = augment.corrupt(mt, c, p)
        info = {"ball_visible": float(np.isfinite(Bm[period > 0, 0]).mean())}
        Pm, Bm = Pm.astype(np.float64), Bm.astype(np.float64)
    else:
        Pm = players.astype(np.float64) * scale
        Bm = ball.astype(np.float64) * scale
        team = np.where(np.isfinite(Pm[..., 0]), slot_team[None, :], -1).astype(np.int8)
    if mirror:
        Pm[..., 1] = W - Pm[..., 1]
        Bm[:, 1] = W - Bm[:, 1]
    info.update(swap=bool(swap), mirror=bool(mirror), kind=kind, seed=int(c))
    return Pm, team, Bm, Y, period, info, _params_dict(p)


def write_copy(root, name, g, s, Pm, team, Bm, Y, period, pitch, info, params, t0):
    X, names, _ = features.build(Pm, team, Bm, period, pitch, features.ALL_RADII)
    v, dmin, near, p = features.series(Pm, team, Bm, period)
    base = stem(root, name, g, s)
    X16 = X.astype(np.float16)
    ok = period > 0
    Z = np.asarray(X16[ok], dtype=np.float64)
    fin = np.isfinite(Z)
    Z = np.where(fin, Z, 0.0)
    np.savez(str(base) + ".st.npz", n=fin.sum(0).astype(np.float64), s1=Z.sum(0), s2=(Z * Z).sum(0))
    np.save(str(base) + ".x.npy", X16)
    np.save(str(base) + ".y.npy", np.stack([Y[k] for k in metric.REFS], 1).astype(np.int8))
    np.save(str(base) + ".b.npy", baseline.codes_of(Pm, team, Bm))
    np.save(str(base) + ".f.npy", np.stack([v, dmin], 1).astype(np.float32))
    np.save(str(base) + ".i.npy", np.stack([near, p], 1).astype(np.int8))
    np.save(str(base) + ".per.npy", period)
    info = dict(info, seconds=time.time() - t0)
    Path(str(base) + ".params.json").write_text(json.dumps({"info": info, "params": params}, default=float))


def load_refs(root, name):
    with open(Path(root) / "refs_{}.pkl".format(name), "rb") as fh:
        return pickle.load(fh)


def _game(root, name):
    if name not in _GAMES:
        _GAMES.clear()                # one match at a time per worker: the arrays are large
        _GAMES[name] = metric.load_game(name)
    return _GAMES[name]


def _job(args):
    root, name, g, s, fallback = args
    done = Path(str(stem(root, name, g, s)) + ".params.json")
    if done.is_file():
        if json.loads(done.read_text())["info"]["kind"] != kind_of(g, s, fallback):
            raise RuntimeError("{}: a copy of the other generator is in the bank".format(done))
        return name, g, s, 0.0
    t0 = time.time()
    game = _game(root, name)
    ref = load_refs(root, name)
    out = make_copy(name, g, s, game, ref["Y"], ref["log"], fallback)
    Pm, team, Bm, Y, period, info, params = out
    write_copy(root, name, g, s, Pm, team, Bm, Y, period, game["pitch_m"], info, params, t0)
    return name, g, s, time.time() - t0


def build(root, jobs, workers, fallback=False):
    args = [(str(root), n, g, s, fallback) for n, g, s in sorted(jobs)]
    t0 = time.time()
    if workers <= 1:
        res = [_job(a) for a in args]
    else:
        with mp.get_context("fork" if sys.platform.startswith("linux") else "spawn").Pool(workers) as pool:
            res = []
            for r in pool.imap_unordered(_job, args):
                res.append(r)
                print("  copy", *r, flush=True)
    print("built {} copies in {:.0f} s".format(len(res), time.time() - t0), flush=True)
    return res


def write_refs(root, ref_params):
    Path(root).mkdir(parents=True, exist_ok=True)
    info = {}
    for name in metric.GAMES_TRAIN:
        G = metric.load_game(name)
        Y, log, inf = metric.references(name, G, ref_params)
        with open(Path(root) / "refs_{}.pkl".format(name), "wb") as fh:
            pickle.dump({"Y": Y, "log": log, "periods": G["periods"], "pitch": G["pitch_m"]}, fh)
        info[name] = inf
    return info


def gate(root, jobs):
    """Gate G: median |hidden - h*| <= 0.02, >= 90 % within 0.03, and the median time of a copy."""
    errs, secs = [], []
    for n, g, s in jobs:
        d = json.loads(Path(str(stem(root, n, g, s)) + ".params.json").read_text())["info"]
        errs.append(d["hidden_err"])
        secs.append(d["seconds"])
    errs = np.array(errs)
    out = {"n": len(errs), "median_err": float(np.median(errs)), "share_in_003": float((errs <= 0.03).mean()),
           "median_seconds": float(np.median(secs)), "max_err": float(errs.max())}
    out["pass"] = bool(out["median_err"] <= 0.02 and out["share_in_003"] >= 0.9
                       and out["median_seconds"] <= MAX_COPY_SECONDS)
    return out


def jobs_of(spec):
    """"names:g:s0-s1,..." -> [(name, g, s)]."""
    out = []
    for item in spec.split(","):
        names, g, ss = item.split(":")
        names = metric.GAMES_TRAIN if names == "all" else tuple(names.split("+"))
        lo, _, hi = ss.partition("-")
        for n in names:
            out += [(n, int(g), s) for s in range(int(lo), int(hi or lo) + 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refs", "build", "gate"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--spec", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-lag", type=int)
    ap.add_argument("--cluster", type=int)
    ap.add_argument("--fallback", action="store_true")
    a = ap.parse_args()
    if a.cmd == "refs":
        print(json.dumps(write_refs(a.root, reference.RefParams(out_lag=a.out_lag, cluster=a.cluster))))
    elif a.cmd == "build":
        build(a.root, jobs_of(a.spec), a.workers, a.fallback)
    else:
        print(json.dumps(gate(a.root, jobs_of(a.spec))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
