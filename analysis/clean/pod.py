"""The steps of the pod, one command each; every decision is taken here from the score tables.

    python -m analysis.clean.pod p2|p3|p3b|p4|p4eval|p5|p6|p7|p8|p9|p10 --work DIR [--price P --spent S]

DIR holds bank/, runs/, state.json (values chosen so far) and the reports. Fix: M1a's out_lag and
cluster are read from analysis/clean/m1a.json and asserted here. Workers come from cpu.max.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from analysis.clean import bands, bank, baseline, c2, forecast, gates, grid, metric, plateau, priors
from mtah.external import metrica_open, reference
from mtah.transfer import features
from mtah.transfer_mt import decode

ROOT = Path(__file__).resolve().parents[2]
UNITS = metric.UNIT_NAMES
P3B_FOLDS = ("metrica1", "J03WMX", "J03WN1")
CENTRE_MIN = forecast.PRIOR_MIN
PARALLEL = 3
G_SEEDS = 6
# table 3.3 scaled to the ceiling of $7 (Nikita, 2026-11-04): cuts above 6.65, stop above 7
CEILING = 7.0
CUT = 6.65


def cpu_quota():
    """vCPU by cgroup cpu.max: nproc shows the host's cores. MTAH_VCPU when cpu.max says nothing."""
    try:
        q, per = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            return max(1, int(int(q) / int(per)))
    except OSError:
        pass
    return int(os.environ["MTAH_VCPU"])


def state(a):
    path = Path(a.work) / "state.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def save_state(a, s):
    (Path(a.work) / "state.json").write_text(json.dumps(s, indent=1, default=float))


def report(a, name, body):
    (Path(a.work) / "{}.json".format(name)).write_text(json.dumps(body, indent=1, default=float))
    print(name, json.dumps(body, default=float)[:3000], flush=True)


def jobs_list(spec):
    return bank.jobs_of(spec)


def cmd_train(a, hold, variant, seed, cfg, tag=""):
    out = str(Path(a.work) / "runs")
    return [sys.executable, str(ROOT / "training" / "train_clean.py"), "--root", str(Path(a.work) / "bank"),
            "--out", out, "--hold", hold, "--variant", variant, "--seed", str(seed), "--tag", tag,
            "--steps", str(cfg["steps"]), "--channels", str(cfg["channels"]), "--blocks", str(cfg["blocks"]),
            "--lr", str(cfg["lr"]), "--dropout", str(cfg["dropout"]), "--weight-decay", str(cfg["wd"]),
            "--batch", str(cfg["batch"]), "--chunk", str(cfg["chunk"]), "--r0", str(cfg["r0"]),
            "--device", a.device, "--g-seeds", str(G_SEEDS)]


def parallel(cmds):
    """Run the commands, PARALLEL at a time, OMP = 3 threads; stop on the first failure."""
    env = dict(os.environ, OMP_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3", MKL_NUM_THREADS="3")
    pending, running = list(cmds), []
    while pending or running:
        while pending and len(running) < PARALLEL:
            running.append(subprocess.Popen(pending.pop(0), env=env))
        time.sleep(1)
        for p in list(running):
            code = p.poll()
            if code is not None:
                running.remove(p)
                if code:
                    for q in running:
                        q.kill()
                    raise RuntimeError("training failed: {}".format(p.args))


def times(a):
    path = Path(a.work) / "runs" / "times.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.is_file() else []


# ------------------------------------------------------------------------------------ P2, P3

def p2(a):
    fixed = json.loads((ROOT / "analysis" / "clean" / "m1a.json").read_text())
    games = [metrica_open.load(g) for g in metric.METRICA]
    lag, n_lag = bands.out_lag_value(games)
    clu, n_clu = bands.cluster_value(games)
    assert lag == fixed["out_lag"] and clu == fixed["cluster"], (lag, clu, fixed)
    rp = reference.RefParams(out_lag=lag, cluster=clu)
    root = str(Path(a.work) / "bank")
    info = bank.write_refs(root, rp)
    rows = []
    for name in metric.GAMES_TRAIN:
        r = bank.load_refs(root, name)
        rows += [bands.stats(r["Y"][k], r["periods"]) for k in metric.REFS]
    b1 = bands.band_b1(rows)
    t0 = time.time()
    if a.fallback:
        for p in Path(root).glob("*_g*_s*.*"):       # copies of G: the bank is deterministic by seed
            p.unlink()
        smoke, gate_jobs = bank.jobs_of("metrica1+J03WMX:1:0"), []
    else:
        smoke = bank.jobs_of("all:1:0")
        gate_jobs = bank.jobs_of("metrica1+J03WMX+J03WR9:1:0-5")
    bank.build(root, sorted(set(smoke + gate_jobs)), cpu_quota(), a.fallback)
    if a.fallback:
        base = str(bank.stem(root, "metrica1", 1, 0))
        names = features.names_bank()
        seen = np.load(base + ".x.npy", mmap_mode="r")[:, names.index("ball_present")]
        y = np.load(base + ".y.npy", mmap_mode="r")[:, 0]
        share = float(np.nanmean(np.asarray(seen, dtype=np.float32)[np.asarray(y) == 2]))
        g = {"pass": share > 0, "ball_seen_on_D_metrica1": share, "fallback": True}
    else:
        g = bank.gate(root, gate_jobs)
    values = {"out_lag": lag, "cluster": clu, "n_lag": n_lag, "n_clu": n_clu}
    (Path(a.work) / "values.json").write_text(json.dumps(values))
    report(a, "p2", dict(values, lag_info=info, B1=b1, gate_G=g, seconds=time.time() - t0))
    s = state(a)
    s.update(out_lag=lag, cluster=clu, B1=b1, gate_G=g, bank_copy_s=g.get("median_seconds"), fallback=a.fallback)
    save_state(a, s)


def p3(a):
    root = str(Path(a.work) / "bank")
    g_jobs = bank.jobs_of("all:1:0-5")
    evals = "+".join(metric.GAMES_EVAL)
    t1 = bank.jobs_of("{}:2:0-1".format(evals))
    t2 = bank.jobs_of("{}:3:0-1".format(evals))
    t0 = time.time()
    s = state(a)
    bank.build(root, g_jobs + t1 + t2, cpu_quota(), s.get("fallback", False))
    g = {"pass": True, "fallback": True} if s.get("fallback") else bank.gate(root, g_jobs + t1)
    s["gate_bank"] = g
    save_state(a, s)
    report(a, "p3", {"gate_G": g, "n": len(g_jobs + t1 + t2), "seconds": time.time() - t0})


# ----------------------------------------------------------------------------------------- P3b

def _config_tag(i):
    return "_c{:02d}".format(i)


def _score_config(a, tag, folds):
    """Maximum over the grid of S0 of the summary on these folds (Metrica 1/2, IDSSE 1/2)."""
    root = str(Path(a.work) / "bank")
    oof = grid.Oof(str(Path(a.work) / "runs"), "S0", [0], tag)
    cells = grid.all_cells(grid.base_axes(False))
    jobs = [(root, oof, n, g, s, cells) for n in folds for g, s in grid.COPIES]
    with grid.pool(cpu_quota()) as p:
        res = list(p.imap_unordered(grid._score_job, jobs))
    by = {}
    for name, g, s, S, _ in res:
        by.setdefault(name, []).append(S.mean(1))
    m = [np.mean(by[n], axis=0) for n in folds if n in metric.METRICA]
    i = [np.mean(by[n], axis=0) for n in folds if n not in metric.METRICA]
    return float((0.5 * np.mean(m, axis=0) + 0.5 * np.mean(i, axis=0)).max())


def p3b(a):
    s = state(a)
    if not s["gate_bank"]["pass"]:
        raise SystemExit("gate G failed on the bank: section 0.4")
    drop = [f for f in a.drop_factors.split(",") if f]
    configs = [c for c in forecast.configs()
               if not any(c[f] != forecast.CENTRE[f] for f in drop)]
    done = {}
    folds = tuple(s.get("p3b_folds", P3B_FOLDS))
    plateau_eps = s.get("p3b_eps", 0.025)
    t_rows = {}
    spent0, t_start = a.spent, time.time()

    def spent():
        return spent0 + (time.time() - t_start) / 3600.0 * a.price

    vcpu = cpu_quota()
    for i, cfg in enumerate(configs):
        tag = _config_tag(i)
        if not all((Path(a.work) / "runs" / "oof_S0{}_{}_s0.npz".format(tag, f)).is_file() for f in folds):
            parallel([cmd_train(a, f, "S0", 0, cfg, tag) for f in folds])
        mins = [t["seconds"] / 60.0 for t in times(a) if tag + "_" in t["tag"]]
        t_rows[i] = float(np.mean(mins))
        done[i] = _score_config(a, tag, folds)
        print("config", i, cfg, "score", done[i], "minutes", t_rows[i], flush=True)
        if i == 0:
            t_centre = t_rows[0]
            n_left = sum(forecast.N_LEFT.values())
            f0 = forecast.forecast(spent(), cfg, t_centre, {}, forecast.N_LEFT,
                                   (len(configs) - 1) * len(folds) * t_centre, a.price, vcpu)
            print("forecast after the centre", f0, "n_left", n_left, flush=True)
            if f0 > CEILING:
                raise SystemExit("forecast above {}: stop, table 3.3".format(CEILING))
            if f0 > CUT and len(folds) == 3:
                folds = ("metrica1", "J03WMX")
                plateau_eps = 0.028
                s.update(p3b_folds=list(folds), p3b_eps=plateau_eps)
                save_state(a, s)
                done[0] = _score_config(a, tag, folds)
                print("forecast above {}: one match per source, plateau 0.028".format(CUT), flush=True)
    scores, t_factor = {}, {}
    for i, cfg in enumerate(configs):
        changed = [f for f in forecast.ORDER if cfg[f] != forecast.CENTRE[f]]
        if changed:
            scores[(changed[0], cfg[changed[0]])] = done[i]
            t_factor[(changed[0], cfg[changed[0]])] = t_rows[i]
    for f in forecast.ORDER:
        scores[(f, forecast.CENTRE[f])] = done[0]
        t_factor[(f, forecast.CENTRE[f])] = t_rows[0]
    fixed = forecast.dropped_values(drop, forecast.CENTRE["blocks"], forecast.CENTRE["batch"])
    chosen, table = forecast.choose(scores, t_rows[0], t_factor, spent(), a.price, vcpu, plateau_eps, fixed)
    s.update(train=chosen, t_centre=t_rows[0], t_factor={"{}={}".format(*k): v for k, v in t_factor.items()})
    save_state(a, s)
    report(a, "p3b", {"chosen": chosen, "table": table, "scores": {str(k): v for k, v in scores.items()},
                      "minutes": t_rows, "folds": folds})


# ----------------------------------------------------------------------------------------- P4

def _train_model(a, variant):
    s = state(a)
    cmds = []
    for u in UNITS:
        for seed in (0, 1):
            if not (Path(a.work) / "runs" / "oof_{}_{}_s{}.npz".format(variant, u, seed)).is_file():
                cmds.append(cmd_train(a, u, variant, seed, s["train"]))
    parallel(cmds)


def p4(a):
    _train_model(a, "S0")


def _table_path(a, variant):
    return Path(a.work) / "table_{}.npz".format(variant)


def score_cells(a, variant, cells):
    """The table of a model on at least these cells."""
    path = _table_path(a, variant)
    t = grid.Table.load(path) if path.is_file() else grid.Table()
    new = t.missing(cells)
    if new:
        oof = grid.Oof(str(Path(a.work) / "runs"), variant, [0, 1])
        S, F = grid.score(str(Path(a.work) / "bank"), oof, new, cpu_quota())
        t.add(new, S, F)
        t.save(path)
    return t


def decoder_of(a, variant, eps_s):
    """Steps 1-6 of section 4.5 for one model: final axes, its cell, nested picks."""
    c1 = "C1" in variant
    axes = grid.base_axes(c1)
    t = score_cells(a, variant, grid.all_cells(axes))
    first = plateau.select(plateau.Lattice(axes, t), UNITS, eps_s)
    ax2, changed = plateau.extended_axes(plateau.Lattice(axes, t), first["idx"])
    if changed:
        t = score_cells(a, variant, grid.all_cells(ax2))
    lat = plateau.Lattice(ax2, t)
    final = plateau.select(lat, UNITS, eps_s)
    inner = plateau.nested(lat, eps_s)
    picks = {u: grid.canon(*r["value"]) for u, r in inner.items()}
    in_set = sum(bool(final["set"][r["idx"]]) for r in inner.values())
    out = {"cell": grid.canon(*final["value"]), "axes": ax2, "extended": changed, "notes": final["notes"],
           "P_scale": final["P_scale"], "n_set": final["n_set"], "n_Q": final["n_Q"], "r_star": final["r_star"],
           "picks": {u: list(v) for u, v in picks.items()}, "in_set": in_set,
           "decoder_unstable": in_set < 5, "S_star": final["S_star"], "diag_width": final["diag_width"],
           "nested_notes": {u: r["notes"] for u, r in inner.items()}}
    return out, t, picks, lat


def _seed_units(a, key):
    """(s_u, f_u) of S0 on one seed at one cell."""
    root = str(Path(a.work) / "bank")
    out = []
    for seed in (0, 1):
        oof = grid.Oof(str(Path(a.work) / "runs"), "S0", [seed])
        S, F = grid.score(root, oof, [key], cpu_quota())
        out.append((grid.unit_scores(S[0]).mean((1, 2)), grid.unit_scores(F[0]).mean((1, 2))))
    return out


def p4eval(a):
    s = state(a)
    cells = grid.all_cells(grid.base_axes(False))
    t = score_cells(a, "S0", cells)
    lat = plateau.Lattice(grid.base_axes(False), t)
    W = plateau.weights(UNITS)
    unit = grid.unit_scores(np.moveaxis(t.S.mean((2, 3)), 1, 0))          # [6, cells]
    star = t.cells[int(np.argmax((W[:, None] * unit).sum(0)))]
    (s0a, f0a), (s0b, f0b) = _seed_units(a, star)
    eps_s, eps_m, daa_s, daa_m = gates.noise(s0a, s0b, f0a, f0b)
    dec, t, picks, lat = decoder_of(a, "S0", eps_s)
    s0 = gates.nested_scores(t, picks)
    base = grid.score_baseline(str(Path(a.work) / "bank"), cpu_quota())
    base_nested = {w: gates.baseline_nested(*base[w]) for w in base}
    base_w = {w: gates.W(n.s) for w, n in base_nested.items()}
    bstar = max(base_w, key=base_w.get)
    g0 = gates.g0(s0, base_nested[bstar])
    rms = math.sqrt(float(np.mean(np.array(daa_s) ** 2)))
    s.update(eps_s=eps_s, eps_m=eps_m, daa_s=daa_s, daa_m=daa_m, rms_daa=rms, base_star=bstar, base_w=base_w,
             S0=dec, G0=g0)
    save_state(a, s)
    np.savez(Path(a.work) / "base_star.npz", S=base[bstar][0], F=base[bstar][1])
    report(a, "p4", {"S_star_cell": star, "eps_s": eps_s, "eps_m": eps_m, "daa_s": daa_s, "daa_m": daa_m,
                     "base_star": bstar, "base_w": base_w, "G0": g0, "S0": dec, "W_S0": gates.W(s0.s),
                     "s_u": s0.s.tolist(), "f_u": s0.f.tolist()})


# ------------------------------------------------------------------------------------ P5 - P7

def p5(a):
    if not state(a)["G0"]["pass"]:
        raise SystemExit("G0 failed: stop, section 9")
    for v in a.variants.split(","):
        _train_model(a, v)


def _single_d(a, variant, cell):
    """Frames of a single D between H and A at this cell, over all scored copies."""
    oof = grid.Oof(str(Path(a.work) / "runs"), variant, [0, 1])
    root = str(Path(a.work) / "bank")
    n = 0
    for name in metric.GAMES_EVAL:
        for g, s in grid.COPIES:
            _, period = grid._gt(root, name, g, s)
            P, b = oof.get(name, g, s)
            lab = decode.decode_periods(P, period, cell[0], cell[1], cell[2], b, cell[3])
            n += int(((lab[1:-1] == "D") & np.isin(lab[:-2], ("H", "A")) & np.isin(lab[2:], ("H", "A"))
                      & (lab[:-2] != lab[2:])).sum())
    return n


def p6(a):
    s = state(a)
    s0_t = grid.Table.load(_table_path(a, "S0"))
    s0 = gates.nested_scores(s0_t, {u: tuple(v) for u, v in s["S0"]["picks"].items()})
    cands, decs = {}, {}
    for v in a.variants.split(","):
        dec, t, picks, _ = decoder_of(a, v, s["eps_s"])
        cands[v] = gates.nested_scores(t, picks)
        decs[v] = dec
    rep = gates.judge(cands, s0, s["eps_s"], s["eps_m"])
    accepted = sorted(n for n, r in rep.items() if r["accepted"])
    single_d = {v: _single_d(a, v, tuple(decs[v]["cell"])) for v in decs}
    single_d["S0"] = _single_d(a, "S0", tuple(s["S0"]["cell"]))
    s.update(decoders=decs, gate_report=rep, accepted=accepted)
    s["M_star"] = accepted[0] if len(accepted) == 1 else "S0" if not accepted else None
    save_state(a, s)
    report(a, "p6", {"report": rep, "accepted": accepted, "decoders": decs, "single_D": single_d})


def p7(a):
    s = state(a)
    acc = s["accepted"]
    if len(acc) < 2:
        print("p7: fewer than 2 accepted, nothing to run", flush=True)
        return
    if a.no_comb:                                         # table 3.3
        s["M_star"] = gates.best_single(s["gate_report"], acc, s["eps_s"], s["eps_m"])
        save_state(a, s)
        return
    v = "+".join(acc)
    _train_model(a, v)
    dec, t, picks, _ = decoder_of(a, v, s["eps_s"])
    comb = gates.nested_scores(t, picks)
    s0 = gates.nested_scores(grid.Table.load(_table_path(a, "S0")), {u: tuple(x) for u, x in s["S0"]["picks"].items()})
    singles = {n: gates.nested_scores(grid.Table.load(_table_path(a, n)),
                                      {u: tuple(x) for u, x in s["decoders"][n]["picks"].items()}) for n in acc}
    kept, info = gates.combination(comb, singles, s0, s["gate_report"], s["eps_s"], s["eps_m"])
    s["decoders"][v] = dec
    s["M_star"] = v if kept == "comb" else kept
    save_state(a, s)
    report(a, "p7", {"kept": s["M_star"], "info": info})


# ----------------------------------------------------------------------------------------- P8

def _transfer_score(a, variant, source, cell):
    """Mean s over the other source's matches, model trained on `source`, seed 0."""
    tag = "src{}".format(source)
    path = Path(a.work) / "runs" / "oof_{}_{}_s0.npz".format(variant, tag)
    root = str(Path(a.work) / "bank")
    names = metric.METRICA if source == "I" else metric.GAMES_EVAL[3:]
    out = []
    with np.load(path) as z:
        for n in names:
            for g, sd in grid.COPIES:
                key = "{}_g{}_s{}".format(n, g, sd)
                y, period = grid._gt(root, n, g, sd)
                per = grid.periods_of(period)
                P = z[key + "/P"].astype(np.float64)
                b = 1.0 / (1.0 + np.exp(-z[key + "/sw"].astype(np.float64))) if key + "/sw" in z.files else None
                lab = decode.decode_periods(P, period, cell[0], cell[1], cell[2], b, cell[3])
                out.append(np.mean([metric.spell_macro(reference.to_chars(y[:, e]), lab, per)[0]
                                    for e in range(y.shape[1])]))
    return float(np.mean(out))


def _source_cell(a, variant, source, eps_s):
    s = state(a)
    dec = s["S0"] if variant == "S0" else s["decoders"][variant]
    t = grid.Table.load(_table_path(a, variant))
    lat = plateau.Lattice(dec["axes"], t)
    return grid.canon(*plateau.select(lat, UNITS, eps_s, source=source)["value"])


def p8_train(a):
    s = state(a)
    star = s["M_star"]
    models = [star] + ([] if star == "S0" else ["S0"])
    cmds = []
    for v in models:
        for src in ("src:I", "src:M"):
            cmds.append(cmd_train(a, src, v, 0, s["train"]))
    parallel(cmds)


def _t3_prob(a, variant, name, g, sd, cfg):
    from mtah.seqmodel import net as seqnet
    from mtah.transfer_mt import net as mtnet
    unit = next(u for u, ms in metric.UNITS.items() if name in ms)
    base = "{}/bank/{}_g{}_s{}".format(a.work, name, g, sd)
    X = np.load(base + ".x.npy", mmap_mode="r")
    period = np.load(base + ".per.npy")
    cols = features.bank_columns(cfg["r0"])
    Xm = np.asarray(X[:, cols], dtype=np.float64)
    names = features.names_rungs()
    if "C2" in variant:
        f, i = np.load(base + ".f.npy"), np.load(base + ".i.npy")
        val = json.loads((Path(a.work) / "runs" / "c2_{}_{}_s0.json".format(variant, unit)).read_text())
        Xm = np.concatenate([Xm, features.event_columns(f[:, 0], f[:, 1], i[:, 0], i[:, 1], period, val)], 1)
        names = names + list(features.EVENT_COLUMNS)
    ws = [seqnet.load(Path(a.work) / "runs" / "w_{}_{}_s{}.npz".format(variant, unit, seed)) for seed in (0, 1)]
    P, aux = mtnet.ours(ws, Xm, names, period)
    return P, (1.0 / (1.0 + np.exp(-aux[:, 0])) if "C1" in variant else None), period


def _t3_job(args):
    a, variant, name, g, sd, cfg, cell = args
    P, b, period = _t3_prob(a, variant, name, g, sd, cfg)
    y = np.load("{}/bank/{}_g{}_s{}.y.npy".format(a.work, name, g, sd))
    per = grid.periods_of(period)
    lab = decode.decode_periods(P, period, cell[0], cell[1], cell[2], b, cell[3])
    return variant, name, sd, float(np.mean([metric.spell_macro(reference.to_chars(y[:, e]), lab, per)[0]
                                             for e in range(y.shape[1])]))


def _base_t3(args):
    a, name, g, sd, w = args
    y, period = grid._gt("{}/bank".format(a.work), name, g, sd)
    per = grid.periods_of(period)
    lab = baseline.labels_of(np.load("{}/bank/{}_g{}_s{}.b.npy".format(a.work, name, g, sd)), per, w)
    return "base", name, sd, float(np.mean([metric.spell_macro(reference.to_chars(y[:, e]), lab, per)[0]
                                            for e in range(y.shape[1])]))


def p8(a):
    s = state(a)
    star = s["M_star"]
    eps_s = s["eps_s"]
    eM = plateau.eps_source(eps_s, UNITS, "M")
    eI = plateau.eps_source(eps_s, UNITS, "I")
    rep = {"M_star_in": star, "transfer": {}, "transfer_removed": bool(a.no_transfer)}
    keep = True
    if not a.no_transfer:
        models = [star] + ([] if star == "S0" else ["S0"])
        tr = {}
        for v in models:
            tr[v] = {}
            for src in ("I", "M"):
                cell = _source_cell(a, v, src, eps_s)
                tr[v][src] = {"cell": cell, "score": _transfer_score(a, v, src, cell)}
        rep["transfer"] = tr
        base = np.load(Path(a.work) / "base_star.npz")["S"].mean((1, 2))        # [9], matches of GAMES_EVAL
        rep["base_star_transfer"] = {"M": float(base[:3].mean()), "I": float(base[3:].mean())}
        rep["S0_below_base"] = {src: tr["S0"][src]["score"] < rep["base_star_transfer"]["M" if src == "I" else "I"]
                                for src in ("I", "M")} if star == "S0" else None
        c2_file = Path(a.work) / "runs" / "c2_{}_srcM_s0.json".format(star)
        if c2_file.is_file():
            rep["c2_on_metrica"] = json.loads(c2_file.read_text())
            rep["c2_on_idsse"] = c2.values(str(Path(a.work) / "bank"), "all", range(G_SEEDS))
        if star != "S0":
            for src, eps in (("I", eM), ("M", eI)):           # trained on src, read on the other
                if tr[star][src]["score"] < tr["S0"][src]["score"] - eps:
                    keep = False
    if not keep:
        star_after = "S0"
    else:
        star_after = star
    # T3: read once, after all decisions
    root = str(Path(a.work) / "bank")
    if not all(Path(str(bank.stem(root, n, 4, sd)) + ".params.json").is_file()
               for n in metric.GAMES_EVAL for sd in range(4)):
        bank.build(root, bank.jobs_of("{}:4:0-3".format("+".join(metric.GAMES_EVAL))), cpu_quota(), s.get("fallback", False))
    cfg = s["train"]
    cells = {"S0": tuple(s["S0"]["cell"])}
    if star_after != "S0":
        cells[star_after] = tuple(s["decoders"][star_after]["cell"])
    jobs = [(a, v, n, 4, sd, cfg, c) for v, c in cells.items() for n in metric.GAMES_EVAL for sd in range(4)]
    bjobs = [(a, n, 4, sd, s["base_star"]) for n in metric.GAMES_EVAL for sd in range(4)]
    with grid.pool(cpu_quota()) as p:
        res = list(p.imap_unordered(_t3_job, jobs)) + list(p.imap_unordered(_base_t3, bjobs))
    by = {}
    for v, n, sd, x in res:
        by.setdefault(v, {}).setdefault(n, []).append(x)
    match = {v: {n: float(np.mean(x)) for n, x in d.items()} for v, d in by.items()}

    def W_of(d):
        return metric.W({u: float(np.mean([d[n] for n in ms])) for u, ms in metric.UNITS.items()})
    rep["T3"] = {"match": match, "W": {v: W_of(d) for v, d in match.items()}}
    if star_after != "S0":
        thr = max(0.04, 2.5 * s["rms_daa"])
        delta = {n: match[star_after][n] - match["S0"][n] for n in metric.GAMES_EVAL}
        wd = W_of(delta)
        if wd < -eps_s or min(delta.values()) < -thr:
            star_after = "S0"
        rep["T3"].update(W_delta=wd, worst_delta=min(delta.values()), threshold=thr)
    rep["M_star_out"] = star_after
    s["M_star"] = star_after
    save_state(a, s)
    report(a, "p8", rep)


# ---------------------------------------------------------------------------------- P9, P10

def p9(a):
    s = state(a)
    star = s["M_star"]
    parallel([cmd_train(a, "all", star, seed, s["train"]) for seed in (0, 1, 2)])


def _b2_job(args):
    a, variant, name, g, sd, cell = args
    oof = grid.Oof(str(Path(a.work) / "runs"), variant, [0, 1])
    _, period = grid._gt("{}/bank".format(a.work), name, g, sd)
    P, b = oof.get(name, g, sd)
    lab = decode.decode_periods(P, period, cell[0], cell[1], cell[2], b, cell[3])
    codes = np.where(lab == "H", 0, np.where(lab == "A", 1, np.where(lab == "D", 2, -1))).astype(np.int8)
    return bands.stats(codes, grid.periods_of(period))


def _a10_job(args):
    a, variant, name, g, sd, cell = args
    _, period = grid._gt("{}/bank".format(a.work), name, g, sd)
    lab = []
    for seed in (0, 1):
        P, b = grid.Oof(str(Path(a.work) / "runs"), variant, [seed]).get(name, g, sd)
        lab.append(decode.decode_periods(P, period, cell[0], cell[1], cell[2], b, cell[3]))
    return float((lab[0] != lab[1])[period > 0].mean())


def p10_bands(a):
    """B2, A10, quantiles of A11; the registry without A6."""
    s = state(a)
    star = s["M_star"]
    dec = s["S0"] if star == "S0" else s["decoders"][star]
    picks = dec["picks"]
    jobs = []
    for u, ms in metric.UNITS.items():
        for n in ms:
            for g, sd in grid.COPIES:
                jobs.append((a, star, n, g, sd, tuple(picks[u])))
    with grid.pool(cpu_quota()) as p:
        rows = list(p.imap_unordered(_b2_job, jobs))
        a10 = list(p.imap_unordered(_a10_job, [j for j in jobs if j[3] == 2]))
    b2 = bands.band_b1(rows)
    cfg = s["train"]
    cols = features.bank_columns(cfg["r0"])
    root = "{}/bank".format(a.work)
    q_lo, q_hi = [], []
    use_events = "C2" in star
    copies = [(n, g, sd) for n in metric.GAMES_EVAL for g, sd in grid.COPIES]
    Xs = []
    val = None
    if use_events:
        val = c2.values(root, "all", range(6))
    for n, g, sd in copies:
        base = "{}/{}_g{}_s{}".format(root, n, g, sd)
        per = np.load(base + ".per.npy")
        x = np.array(np.load(base + ".x.npy", mmap_mode="r")[per > 0][:, cols], dtype=np.float32)
        if use_events:
            f, i = np.load(base + ".f.npy"), np.load(base + ".i.npy")
            e = features.event_columns(f[:, 0], f[:, 1], i[:, 0], i[:, 1], per, val)[per > 0]
            x = np.concatenate([x, e.astype(np.float16).astype(np.float32)], 1)
        Xs.append(x)
    Z = np.concatenate(Xs)
    for k in range(Z.shape[1]):
        lo, hi = np.nanpercentile(Z[:, k], [0.1, 99.9])
        q_lo.append(float(lo))
        q_hi.append(float(hi))
    names = features.names_rungs() + (list(features.EVENT_COLUMNS) if use_events else [])
    s.update(B2=b2, a10=float(max(a10) * 1.25), q_names=names, q_lo=q_lo, q_hi=q_hi, c2=val)
    save_state(a, s)
    report(a, "p10_bands", {"B2": b2, "a10": s["a10"]})


def registry(a):
    """registry.json and decode.json from state.json, the final weights under weights/."""
    s = state(a)
    star = s["M_star"]
    dec = s["S0"] if star == "S0" else s["decoders"][star]
    parts = star.split("+")
    c = dec["cell"]
    models = {"S0": s["S0"], **s.get("decoders", {})}
    reg = {"model_id": star, "use_boundary": "C1" in parts, "use_events": "C2" in parts, "use_r75": "C7" in parts,
           "decoders": {k: {"switch": v["cell"][0], "dead": v["cell"][1], "kappa": v["cell"][2],
                            "switch_min": v["cell"][3], "decoder_unstable": v["decoder_unstable"]}
                        for k, v in models.items()},
           "train": s["train"], "out_lag": s["out_lag"], "cluster": s["cluster"], "eps_s": s["eps_s"],
           "eps_m": s["eps_m"], "base_star": s["base_star"], "B1": s["B1"], "B2": s["B2"], "A6": s.get("a6"),
           "A10": s["a10"], "quantile_names": s["q_names"], "quantile_lo": s["q_lo"], "quantile_hi": s["q_hi"],
           "priors_md5": priors.table_md5()}
    decode_cfg = {"r0": s["train"]["r0"], "switch": c[0], "dead": c[1], "kappa": c[2], "switch_min": c[3],
                  "use_boundary": reg["use_boundary"], "use_events": reg["use_events"], "model_id": star}
    if reg["use_events"]:
        for k in ("v_still", "T_still", "v_kick", "d_player"):
            reg[k] = s["c2"][k]
            decode_cfg[k] = s["c2"][k]
    (Path(a.work) / "registry.json").write_text(json.dumps(reg, indent=1, default=float))
    (Path(a.work) / "decode.json").write_text(json.dumps(decode_cfg, indent=1, default=float))
    out = Path(a.work) / "weights"
    out.mkdir(exist_ok=True)
    (out / "decode.json").write_text((Path(a.work) / "decode.json").read_text())
    for seed in (0, 1, 2):
        shutil.copy(Path(a.work) / "runs" / "w_{}_all_s{}.npz".format(star, seed), out / "seed{}.npz".format(seed))
    print(hashlib.md5((Path(a.work) / "registry.json").read_bytes()).hexdigest(), "registry.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--work", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--price", type=float, required=True)
    ap.add_argument("--spent", type=float, required=True)
    ap.add_argument("--variants", default="C1,C2,C7")
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--no-comb", action="store_true", help="table 3.3: P7 removed")
    ap.add_argument("--no-transfer", action="store_true", help="table 3.3: transfer removed")
    ap.add_argument("--drop-factors", default="", help="table 3.3: factors of P3b removed")
    a = ap.parse_args()
    a.work = str(Path(a.work).resolve())
    Path(a.work, "bank").mkdir(parents=True, exist_ok=True)
    Path(a.work, "runs").mkdir(parents=True, exist_ok=True)
    steps = {"p2": p2, "p3": p3, "p3b": p3b, "p4": p4, "p4eval": p4eval, "p5": p5, "p6": p6, "p7": p7,
             "p8train": p8_train, "p8": p8, "p9": p9, "p10bands": p10_bands, "registry": registry}
    steps[a.step](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
