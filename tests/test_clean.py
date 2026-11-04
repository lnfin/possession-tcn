"""Tests of step M3. Synthetic unless marked; `data` tests need the open matches on this machine."""

import dataclasses
import importlib
import inspect
import os
import pkgutil  # noqa: F401
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.clean import baseline, c2, forecast, gates, grid, metric, plateau, priors  # noqa: E402
from mtah import data, paths  # noqa: E402
from mtah.external import corrupt, idsse_open, reference  # noqa: E402
from mtah.seqmodel import decode as seqdecode  # noqa: E402
from mtah.sota import idsse  # noqa: E402
from mtah.transfer import augment, features, match, ours  # noqa: E402
from mtah.transfer_dom import simulate  # noqa: E402
from mtah.transfer_mt import decode  # noqa: E402
from mtah.transfer_unified import simulate2  # noqa: E402

HAS_DATA = os.environ.get("MTAH_DATA_DIR") and (Path(os.environ["MTAH_DATA_DIR"]) / "cache" / "external" / "idsse").is_dir()
OFFICIAL = paths.official_repo()
data_needed = pytest.mark.skipif(not HAS_DATA, reason="open matches not on this machine")
MODULES = [".".join(p.relative_to(ROOT).with_suffix("").parts) for p in (ROOT / "mtah").rglob("*.py")
           if p.name != "__init__.py"]
MODULES += ["analysis.clean." + m.name for m in pkgutil.iter_modules([str(ROOT / "analysis" / "clean")])]
MODULES += ["tools.pipeline_check"] if os.environ.get("MTAH_DATA_DIR") else []
MODULES += ["training.tcn", "training.train_clean"]


def test_all_modules_import():
    for name in MODULES:
        importlib.import_module(name)


def _dataclasses():
    out = []
    for name in MODULES:
        mod = importlib.import_module(name)
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if dataclasses.is_dataclass(cls) and cls.__module__.startswith(("mtah", "analysis")):
                out.append(cls)
    return set(out)


def test_no_dataclass_defaults():
    classes = _dataclasses()
    assert {c.__name__ for c in classes} >= {"CorruptParams", "DeadBallParams", "SimParams", "SimParams2",
                                             "AugParams", "RefParams", "BallParams", "Match"}
    for cls in classes:
        for f in dataclasses.fields(cls):
            assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING, (cls, f.name)


ALLOWED_DEFAULTS = {"fps": 25, "cap": None, "max_lag": None, "adjust": 0}   # adjust: official games.py


def test_no_numeric_function_defaults():
    for name in MODULES:
        mod = importlib.import_module(name)
        if "third_party" in mod.__name__:
            continue
        for fname, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn.__module__ != mod.__name__:
                continue
            for p in inspect.signature(fn).parameters.values():
                if isinstance(p.default, (int, float)) and not isinstance(p.default, bool):
                    assert p.name in ALLOWED_DEFAULTS, (name, fname, p.name)


def test_no_default_constructor():
    with pytest.raises(TypeError):
        corrupt.CorruptParams()


# ------------------------------------------------------------------------------------ the reference

def events(rows):
    cols = ["team", "type", "subtypes", "period", "start", "end", "sx", "sy", "ex", "ey"]
    out = []
    for r in rows:
        r = dict(team=0, subtypes=frozenset(), period=1, sx=0.5, sy=0.5, ex=0.5, ey=0.5, **r)
        out.append(r)
    return pd.DataFrame(out)[cols]


RP = reference.RefParams(out_lag=3, cluster=39)


def test_branch_order_goal_before_out():
    class E:
        type, start, end, ex, ey = "BALL OUT", 10, 20, 0.5, 0.5
        subtypes = frozenset({"GOAL"})
    assert reference.stoppage_start(E, RP) == (20, "goal")
    E.subtypes = frozenset()
    assert reference.stoppage_start(E, RP) == (17, "out")
    E.type, E.subtypes = "PASS", frozenset({"OUT"})
    assert reference.stoppage_start(E, RP) == (20, "out")
    E.type, E.subtypes = "FAULT RECEIVED", frozenset()
    assert reference.stoppage_start(E, RP) == (20, "referee")
    E.type, E.subtypes = "CHALLENGE", frozenset({"FAULT"})
    assert reference.stoppage_start(E, RP) == (20, "referee")
    E.type, E.subtypes = "BALL LOST", frozenset({"OFFSIDE"})
    assert reference.stoppage_start(E, RP) == (20, "referee")
    E.type, E.subtypes = "RECOVERY", frozenset()
    assert reference.stoppage_start(E, RP) == (20, "unexplained")


def test_dead_stretch_and_goal_kick_closes():
    rows = [dict(type="PASS", start=10, end=20), dict(type="BALL OUT", start=100, end=110)]
    ev = events(rows)
    ev = pd.concat([ev, pd.DataFrame([dict(team=1, type="PASS", subtypes=frozenset({"GOAL KICK"}), period=1,
                                           start=200, end=210, sx=.5, sy=.5, ex=.5, ey=.5)])], ignore_index=True)
    g = {"ball": np.zeros((1000, 2)), "periods": ((1, 0, 999),), "events": ev}
    y = reference.labels(g, RP)
    assert (y[:10] == 2).all() and (y[107:200] == 2).all() and y[106] == 0 and y[200] == 1 and (y[211:] == 2).all()


def test_versions_keep_D():
    y = np.array([2] * 5 + [0] * 3 + [1] * 2 + [0] * 40 + [2] * 3 + [1] * 2 + [0] * 30, dtype=np.int8)
    per = ((1, 0, len(y) - 1),)
    for L in (25, 75, 100):
        m = reference.merge(y, per, L)
        assert ((m == 2) == (y == 2)).all()
    m = reference.merge(y, per, 25)
    assert (m[5:8] == 0).all(), "a run right after D stays"
    assert (m[8:10] == 0).all(), "a short run joins the previous team label"
    assert (m[53:55] == 1).all()
    d = reference.keep_runs(y, per, lambda a, z, k: False)
    assert ((d == 2) == (y == 2)).all()


def test_lag_ties():
    y = np.zeros(1000, dtype=np.int8)
    lag, share, _, _ = reference.lag_of([(1, 10.0, 0)], y, ((1, 0, 999),))
    assert lag == 0 and share == 1.0
    y = np.ones(1000, dtype=np.int8)
    y[241:246] = 0
    y[255:260] = 0
    assert reference.lag_of([(1, 10.0, 0)], y, ((1, 0, 999),))[0] == -5


def test_folders_equal():
    assert idsse_open.folder() == idsse.folder()


def test_valid_runs_not_used():
    for path in (ROOT / "analysis" / "clean").glob("*.py"):
        assert "pm.valid_runs(" not in path.read_text()


# ----------------------------------------------------------------------------------- the features

def test_feature_columns():
    assert len(features.names_rungs()) == 113 and len(features.names_bank()) == 137
    assert len(features.names_rungs()) + len(features.EVENT_COLUMNS) == 118
    names = features.names_rungs()
    assert names[:7] == ["ball_present", "ball_x", "ball_y", "ball_absy", "ball_speed", "ball_out_dist", "rel_H"]
    assert names[18] == "d_diff" and names[19] == "ctrl_H@0" and names[36] == "ctrl_until@2"
    assert names[37] == "ball_since" and names[45] == "h_n@25" and names[-2:] == ["pspeed1", "pspeed@75"]
    assert names[45 + 28:45 + 33] == ["gap_cx@25", "front_back@25", "back_front@25", "speed_diff@25", "n_centre@25"]
    assert features.EVENT_COLUMNS == ("ev_still", "ev_since_still", "ev_kick_H", "ev_kick_A", "ev_since_kick")
    for r0 in (0.5, 0.75, 1.0):
        idx = features.bank_columns(r0)
        assert len(idx) == 113 and len(set(idx)) == 113


def synth_match(T=4000, seed=0):
    r = np.random.default_rng(seed)
    P = np.cumsum(r.normal(0, 0.1, (T, 22, 2)), axis=0) + r.uniform(30, 70, (1, 22, 2))
    P[:, :11, 0] -= 15
    P[:, 11:, 0] += 15
    team = np.r_[np.zeros(11), np.ones(11)].astype(np.int8)
    B = P[np.arange(T), (np.arange(T) // 100) % 22] + 0.3
    B[1000:1300] = np.nan
    period = np.r_[np.ones(T // 2), 2 * np.ones(T - T // 2)].astype(np.int8)
    return P, team, B, period


def test_features_bank_equals_model_columns():
    P, team, B, period = synth_match()
    pitch = (105.0, 68.0)
    Xb, nb, _ = features.build(P, team, B, period, pitch, features.ALL_RADII)
    Xm, nm, _ = features.build(P, team, B, period, pitch, (0.75, 1.5, 3.0))
    assert nb == features.names_bank() and nm == features.names_rungs()
    np.testing.assert_array_equal(Xb[:, features.bank_columns(0.75)], Xm)


def test_rotation_removed_exactly():
    P, team, B, period = synth_match()
    pitch = (105.0, 68.0)
    a = features.build(P, team, B, period, pitch, (0.75, 1.5, 3.0))[0]
    R, Q = P.copy(), B.copy()
    R[..., 0], R[..., 1] = 105.0 - P[..., 0], 68.0 - P[..., 1]
    Q[..., 0], Q[..., 1] = 105.0 - B[..., 0], 68.0 - B[..., 1]
    b = features.build(R, team, Q, period, pitch, (0.75, 1.5, 3.0))[0]
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_ball_out_dist_sign():
    T = 300
    P = np.full((T, 4, 2), 50.0)
    P[:, 2:, 0] = 60.0
    P[:, 0, 0], P[:, 2, 0] = 30.0, 70.0
    B = np.tile(np.array([[50.0, 30.0]]), (T, 1))
    B[100:200, 0] = 107.0
    period = np.ones(T, dtype=np.int8)
    team = np.array([0, 0, 1, 1], dtype=np.int8)
    X, names, _ = features.build(P, team, B, period, (105.0, 68.0), (0.75, 1.5, 3.0))
    k = names.index("ball_out_dist")
    assert abs(X[150, k] - 0.4) < 1e-5 and X[50, k] < 0


def test_events_columns():
    T = 200
    v = np.full(T, 0.1, dtype=np.float32)
    v[60:] = 12.0
    p = np.ones(T, dtype=np.int8)
    dmin = np.full(T, 0.5, dtype=np.float32)
    near = np.ones(T, dtype=np.int8)
    period = np.ones(T, dtype=np.int8)
    val = {"v_still": 1.0, "T_still": 5, "v_kick": 8.0, "d_player": 1.0}
    E = features.event_columns(v, dmin, near, p, period, val)
    names = list(features.EVENT_COLUMNS)
    assert E[:, names.index("ev_still")][:4].sum() == 0 and E[10, names.index("ev_still")] == 1
    assert E[59, names.index("ev_still")] == 1 and E[60, names.index("ev_still")] == 0
    assert E[60, names.index("ev_kick_A")] == 1 and E[:, names.index("ev_kick_H")].sum() == 0
    p2 = p.copy()
    p2[60] = 0
    assert features.event_columns(v, dmin, near, p2, period, val)[60, names.index("ev_kick_A")] == 0
    assert E[150, names.index("ev_since_kick")] < 1.0
    assert features.event_columns(v * 0 + 12, dmin, near, p, period, val)[100, names.index("ev_since_still")] == 1.0


def test_c2_values():
    v = np.linspace(0.5, 0.1, 400).astype(np.float32)
    v[100:] = 15.0
    dmin = np.full(400, 0.7, dtype=np.float32)
    val = c2.compute([(v, dmin, [100])])
    assert val["T_still"] >= 20 and val["v_kick"] == 15.0 and abs(val["d_player"] - 0.7) < 1e-6


# --------------------------------------------------------------------------------------- decoders

def test_viterbi3_equals_generic():
    r = np.random.default_rng(1)
    cost = -np.log(r.dirichlet((1, 1, 1), 3000))
    a = decode.viterbi3(cost, 16.0, 24.0, 24.0)
    t = np.array([[0.0, 16.0, 24.0], [16.0, 0.0, 24.0], [24.0, 24.0, 0.0]])
    np.testing.assert_array_equal(a, seqdecode.viterbi(cost, t))
    np.testing.assert_array_equal(a, decode.viterbi3(cost, np.full(3000, 16.0), 24.0, 24.0))


def test_no_single_D_between_teams():
    T = 400
    P = np.zeros((T, 3))
    P[:T // 2] = (0.97, 0.02, 0.01)
    P[T // 2:] = (0.02, 0.97, 0.01)
    period = np.ones(T, dtype=np.int8)
    for s, d in ((8, 12), (24, 12), (24, 36), (48, 24)):
        lab = decode.decode_periods(P, period, float(s), float(d), 0.0, None, 0.0)
        assert not (lab == "D").any()


def test_switch_costs():
    b = np.array([1e-9, 0.5, 1.0])
    c = decode.switch_costs(10.0, 2.0, b, 1.0)
    assert c[1] == 10.0 and c[2] == 1.0 and c[0] > 10
    assert decode.switch_costs(10.0, 0.0, b, 1.0) == 10.0


def test_grid_sizes():
    assert len(grid.all_cells(grid.base_axes(False))) == 45
    assert len(grid.all_cells(grid.base_axes(True))) == 45 + 45 * 3 * 5


def table_from(fn, axes):
    cells = grid.all_cells(axes)
    t = grid.Table()
    S = np.zeros((len(cells), 9, 4, 4))
    for i, c in enumerate(cells):
        S[i] = fn(c)
    t.add(cells, S, S.copy())
    return t


def bump(c):
    return 0.7 - 0.0005 * (np.log2(c[0] / 24) ** 2 + np.log2(c[1] / 36) ** 2)


def test_select_inner_cell():
    axes = grid.base_axes(False)
    t = table_from(bump, axes)
    r = plateau.select(plateau.Lattice(axes, t), metric.UNIT_NAMES, 0.0001)
    assert r["value"][:2] == (24.0, 36.0)


def test_shift_rounding_and_bounds():
    axes = grid.base_axes(True)
    t = table_from(bump, axes)
    lat = plateau.Lattice(axes, t)
    idx = (axes[0].index(24.0), axes[1].index(36.0), 1, 2)
    assert lat.value(lat.shifted(idx, 0.5))[:2] == (12.0, 16.0)         # 18 lies between 16 and 24: nearer 16
    assert lat.value(lat.shifted(idx, 2.0))[:2] == (48.0, 64.0)         # 72 -> 64
    assert lat.value(lat.shifted(idx, 2.0))[3] == 4.0 and lat.value(idx)[2] == 2.0
    low = (0, axes[1].index(36.0), 1, 2)
    assert lat.shifted(low, 0.5) is None
    edge = (axes[0].index(24.0), axes[1].index(36.0), 1, 0)
    assert lat.shifted(edge, 0.5) is None


def test_kappa_zero_shares_a_value():
    axes = grid.base_axes(True)
    t = table_from(bump, axes)
    lat = plateau.Lattice(axes, t)
    a = lat.at[2, 2, 0, 1]
    assert a == lat.at[2, 2, 0, 3]
    assert lat.at[2, 2, 1, 1] != lat.at[2, 2, 1, 3]


def test_extension_rule():
    axes = grid.base_axes(False)
    t = table_from(bump, axes)
    lat = plateau.Lattice(axes, t)
    ax, ch = plateau.extended_axes(lat, (axes[0].index(48.0), axes[1].index(36.0)))
    assert ax[0][-1] == 96.0 * 96.0 / 64.0 and len(ax[1]) == 7
    ax, ch = plateau.extended_axes(lat, (axes[0].index(16.0), axes[1].index(24.0)))
    assert abs(ax[0][0] - 8.0 * 8.0 / 12.0) < 1e-9 and ax[1][0] == 12.0 * 12.0 / 16.0
    ax, ch = plateau.extended_axes(lat, (axes[0].index(24.0), axes[1].index(36.0)))
    assert not ch


def test_deepest_counts_outer_edges():
    axes = grid.base_axes(False)
    t = table_from(bump, axes)
    lat = plateau.Lattice(axes, t)
    Q = {(i, j) for i in range(2, 6) for j in range(1, 6)}
    pick = plateau._deepest(lat, Q)
    assert pick == (4, 4)


def test_weights_and_neff():
    w = plateau.weights(metric.UNIT_NAMES)
    assert abs(w.sum() - 1) < 1e-12 and abs(w[5] - 1 / 3) < 1e-12
    assert abs(plateau.n_eff(metric.UNIT_NAMES, "M") - 3) < 1e-9
    assert abs(plateau.eps_source(0.019, metric.UNIT_NAMES, "M") - 0.019 * np.sqrt(1.6)) < 1e-12
    assert abs(plateau.eps_source(0.019, metric.UNIT_NAMES, "I") - 0.0294) < 1e-3
    assert abs(1 / (np.array(list(metric.UNIT_W.values())) ** 2).sum() - 4.8) < 1e-9


def test_signflip():
    d = np.ones(6) * 0.03
    assert gates.signflip_p(d) == 1 / 64
    d[0] = -0.03
    assert gates.signflip_p(d) > 1 / 64


def make_nested(s, shift=0.0):
    s = np.asarray(s, dtype=np.float64) + shift
    return gates.Nested(s, s * 0 + 0.7, np.tile(s[:, None], (1, 4)), s.copy())


def test_gate_decisions():
    s0 = make_nested(np.full(6, 0.5))
    good = make_nested(np.full(6, 0.53))
    bad = make_nested(np.full(6, 0.49))
    rep = gates.judge({"C1": good, "C7": bad}, s0, 0.019, 0.005)
    assert rep["C1"]["accepted"] and not rep["C7"]["accepted"]
    only_i = make_nested(np.r_[np.full(3, 0.5), np.full(3, 0.53)])
    rep = gates.judge({"C2": only_i}, s0, 0.019, 0.005)
    assert not rep["C2"]["accepted"]


def test_tie_route_needs_macro_without_each_unit():
    s0 = make_nested(np.full(6, 0.5))
    wide = make_nested(np.full(6, 0.505))
    wide.f = wide.f + 0.02
    assert gates.judge({"C1": wide}, s0, 0.019, 0.005)["C1"]["accepted"]
    one = make_nested(np.full(6, 0.505))
    one.f = one.f + np.array([0, 0, 0, 0, 0, 0.09])
    assert not gates.judge({"C1": one}, s0, 0.019, 0.005)["C1"]["accepted"]


def test_tie_break_of_singles():
    rep = {"C1": {"T": 0.04, "T_macro": 0.0}, "C7": {"T": 0.03, "T_macro": 0.0}, "C2": {"T": 0.01, "T_macro": 0.0}}
    assert gates.best_single(rep, ["C1", "C7", "C2"], 0.019, 0.005) == "C7"
    rep["C1"]["T_macro"] = 0.02
    assert gates.best_single(rep, ["C1", "C7", "C2"], 0.019, 0.005) == "C1"


def test_dropped_factor_values():
    assert forecast.dropped_values(["wd", "dropout"], 8, 16) == {"wd": 0.0, "dropout": 0.0}
    assert forecast.dropped_values(["chunk"], 8, 16) == {"chunk": 1500}
    assert forecast.dropped_values(["chunk"], 9, 16) == {"chunk": 3000}


def test_g0():
    s0 = make_nested(np.full(6, 0.6))
    assert gates.g0(s0, make_nested(np.full(6, 0.5)))["pass"]
    assert not gates.g0(s0, make_nested(np.full(6, 0.58)))["pass"]


def test_forecast_configs():
    cfg = forecast.configs()
    assert len(cfg) == 19 and cfg[0] == forecast.CENTRE
    t = {("steps", 6000): 8.0}
    assert abs(forecast.time_of(dict(forecast.CENTRE, steps=6000), 4.0, t) - 8.0) < 1e-9


# ------------------------------------------------------------------------------------- generators

def test_t1_bounds():
    assert priors.widen("h", priors.H_) == pytest.approx(("U", 0.0975, 0.7225))
    assert priors.widen("missing", priors.HALF) == ("U", 0.0, 0.5)
    k, a, b = priors.widen("life", priors.DUR)
    assert a == pytest.approx(np.exp(0.5 * np.log(60) - 0.625 * np.log(60)))
    assert priors.widen("ball_lag", priors.LAG) == ("I", 0, 6)
    assert priors.widen("ball_smooth", priors.ODD) == priors.ODD
    assert priors.widen("carry_speed", dict(priors.DEAD)["carry_speed"])[1] == 0.5
    assert priors.widen("out_tau_s", dict(priors.DEAD)["out_tau_s"])[1] == 0.05


def test_draw_stream_and_stress():
    a, b = priors.draw("G", 1000000), priors.draw("G", 1000000)
    assert a == b
    assert priors.draw("G", 1000001) != a
    p = priors.draw("T3a", 4000000)
    assert p.wrong_share == 0.375 and p.h == pytest.approx(0.7225)
    q = priors.draw("T3b", 4000002)
    assert q.u == 1.0 and q.ball_smooth == 31 and q.wrong_team == 0.375
    for s in range(50):
        assert priors.draw("T2", s).ball_smooth % 2 == 1


def test_gap_length_at_least_one_frame():
    rng = np.random.default_rng(0)
    p = priors.draw("T2", 3000000)
    p = dataclasses.replace(p, gap_median_s=0.001, gap_sigma=0.0)
    n = 2000
    miss, _, _ = corrupt._gap_mask(rng, np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.ones(n, dtype=bool),
                                   0.3, 0.3, p, 25)
    assert miss.any()


def synth_game(T=3000, seed=0):
    r = np.random.default_rng(seed)
    P = np.cumsum(r.normal(0, 0.001, (T, 24, 2)), axis=0) + r.uniform(0.2, 0.8, (1, 24, 2))
    ball = P[np.arange(T), (np.arange(T) // 80) % 24].astype(np.float32).copy()
    ball[1000:1200] = np.nan
    y = np.zeros(T, dtype=np.int8)
    y[(np.arange(T) // 200) % 2 == 1] = 1
    y[1000:1200] = 2
    return {"players": P.astype(np.float32), "ball": ball, "slot_team": np.r_[np.zeros(12), np.ones(12)].astype(np.int8),
            "periods": ((1, 0, T // 2 - 1), (2, T // 2, T - 1)), "pitch_m": (105.0, 68.0), "fps": 25}, y


@pytest.mark.parametrize("kind", ["G", "T1", "T2"])
def test_generator_runs_and_is_deterministic(kind):
    game, y = synth_game()
    p = priors.draw(kind, 1000000 + 7)
    fn = corrupt.to_atd if kind == "T2" else simulate2.to_atd
    a = fn(game, y, [], p)
    b = fn(game, y, [], p)
    assert a[0].shape == (len(y), 46, 2) and a[1].shape == (len(y), 2)
    assert (a[2] == np.r_[np.zeros(24), np.ones(22)]).all()
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_fallback_generator():
    game, y = synth_game()
    p = priors.draw("AUG", 1000000)
    m = match.game_to_match(game, y, [], p.dead, np.random.default_rng(0))
    P, team, B, lab = augment.corrupt(m, 0, p)
    assert P.shape == m.players.shape and (lab == m.labels).all() and team.shape == P.shape[:2]


def test_wrong_team_adds_pieces_without_contacts():
    T, S = 2000, 10
    P = np.zeros((T, S, 2))
    P[:, :, 0] = 10.0 * np.arange(S)
    has = np.ones((T, S), dtype=bool)
    slot = np.r_[np.zeros(5), np.ones(5)].astype(np.int8)
    team = np.tile(slot, (T, 1)).astype(np.int8)
    sp = simulate.SimParams(missing_share=0.0, life=10.0, off_sigma=0.5, wrong_share=0.3, wrong_mean_s=1.0, contact_m=1.0)
    h2, t2 = simulate._wrong_team(P, has, team, slot, np.random.RandomState(0), sp)
    assert (h2 == has).all()
    share = (t2 != team).mean()
    assert 0.15 < share < 0.35


def test_hidden_share_reaches_target():
    game, y = synth_game(T=20000)
    p = dataclasses.replace(priors.draw("G", 1000000), h=0.5, gap_median_s=5.0, gap_sigma=0.5)
    _, ball, _, info = simulate2.to_atd(game, y, [], p)
    assert abs(info["hidden"] - 0.5) <= 0.02 and np.isnan(ball[:, 0]).mean() == pytest.approx(info["hidden"], abs=0.02)


def test_to_atd_has_no_speed_column():
    assert not hasattr(corrupt, "speed")


# -------------------------------------------------------------------------------------- baseline

@pytest.mark.skipif(not (OFFICIAL / "starter" / "baseline.py").is_file(), reason="official repo absent")
def test_vector_baseline_equals_official(tmp_path):
    sys.path.insert(0, str(OFFICIAL))
    sys.path.insert(0, str(OFFICIAL / "starter"))
    starter = importlib.import_module("baseline")
    if not hasattr(starter, "nearest_player_labels"):
        spec = importlib.util.spec_from_file_location("official_baseline", OFFICIAL / "starter" / "baseline.py")
        starter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(starter)
    r = np.random.default_rng(3)
    T, S = 400, 6
    P = r.uniform(0, 1, (T, S, 2))
    P[r.uniform(size=(T, S)) < 0.2] = np.nan
    B = r.uniform(0, 1, (T, 2))
    B[r.uniform(size=T) < 0.3] = np.nan
    team = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)

    def n(v):
        return "NaN" if not np.isfinite(v) else "{:.6f}".format(v)
    lines = []
    for t in range(T):
        groups = ";".join("{},{},0".format(n(P[t, s, 0]), n(P[t, s, 1])) for s in range(S))
        lines.append("{}:{}:{},{},0".format(t + 1, groups, n(B[t, 0]), n(B[t, 1])))
    raw = tmp_path / "x.txt"
    raw.write_text("\n".join(lines) + "\n")
    pitch = (105.3, 68.0)
    a = starter.nearest_player_labels(str(raw), team, T, pitch)
    scale = np.array(pitch)
    P2 = np.round(P, 6) * scale
    B2 = np.round(B, 6) * scale
    b = baseline.codes_of(P2, team, B2)
    np.testing.assert_array_equal(a, b)
    for w in (0, 25, 125):
        np.testing.assert_array_equal(starter.smooth(a, w), baseline.smooth(b, w))


# ------------------------------------------------------------------------------------ the match files

def _assets():
    return (Path(os.environ.get("MTAH_DATA_DIR", "/nonexistent")) / "tactical").is_dir()


@pytest.mark.skipif(not _assets(), reason="match files not on this machine")
def test_slot_literals_match_the_xml_structure():
    """Reads only the structure of the XML: the number of slots per team."""
    teams = data.slot_teams()
    assert corrupt.N_SLOTS == len(teams)
    assert int((corrupt.SLOT_TEAM == 0).sum()) == int((teams == 0).sum())
    assert int((corrupt.SLOT_TEAM == 1).sum()) == int((teams == 1).sum())


def test_blocks_against_faq():
    """Frame literals live here only: FAQ.md:35, FAQ.md:38-39."""
    assert ours.blocks() == [(1, 11185, 90651), (2, 110808, 190197)]
    assert ours.play_starts(pd.DataFrame({"code": ["SET PIECES"], "Type": ["Kick Off (Start)"],
                                          "start_frame": [113742]}), ours.blocks()) == {1: 11185, 2: 113742}


def test_l2_takes_earliest_tag_and_defaults_to_first():
    blocks = [(1, 100, 200), (2, 300, 400)]
    tags = pd.DataFrame({"code": ["SET PIECES", "SET PIECES", "SET PIECES"], "start_frame": [120, 110, 350],
                         "Type": ["KICK OFF (START)", "Kick Off (Start)", "Corner Kick"]})
    assert ours.play_starts(tags, blocks) == {1: 110, 2: 300}
    p = ours.play_period(ours.block_of(500, blocks), {1: 110, 2: 300}, blocks)
    assert (p[100:110] == 0).all() and p[110] == 1 and p[300] == 2 and p[99] == 0


def test_games_copy_md5():
    import hashlib
    assert hashlib.md5((ROOT / "mtah" / "third_party" / "games.py").read_bytes()).hexdigest() == \
        "dbb7b689189f831090c72d225b8ac194"


# ---------------------------------------------------------------------------------------- statics

def py_files(*folders):
    out = []
    for f in folders:
        p = ROOT / f
        out += [p] if p.is_file() else [q for q in p.rglob("*.py") if "third_party" not in str(q)]
    return out


def test_grep_frame_literals():
    for path in py_files("mtah", "run.py"):
        hits = re.findall(r"\b[0-9]{5,6}\b", path.read_text())
        assert not hits, (path, hits)


def test_grep_evaluation_traces():
    pat = re.compile(r"validation|review|evaluate|gt_public|public_gt|run_evaluation|baseline_(raw|smooth)|\bw[1-7]\b")
    for path in py_files("mtah", "run.py", "training", "analysis", "tools"):
        if path.name != "pipeline_check.py":
            assert not pat.search(path.read_text()), path


def test_grep_runtime_fit():
    pat = re.compile(r"\.fit\(|LogisticRegression|sklearn")
    for path in py_files("mtah"):
        assert not pat.search(path.read_text()), path


def test_grep_provenance():
    pat = re.compile(r"public (gt|window)|hand.?label|this match|our match|in this referee|ours:|v5's labels|red-team|our ATD", re.I)
    for path in py_files("mtah", "analysis", "tools", "run.py", "training"):
        if path.name != "pipeline_check.py":
            assert not pat.search(path.read_text()), path


@pytest.mark.skipif(not (OFFICIAL / "EVALUATION.md").is_file(), reason="official repo absent")
def test_grep_evaluation_numbers():
    src = (OFFICIAL / "EVALUATION.md").read_text().splitlines()
    tokens = set()
    for a, b in ((107, 110), (132, 133)):
        for line in src[a - 1:b]:
            tokens |= set(re.findall(r"\b0\.\d{4}\b", line))
    files = py_files("mtah", "analysis", "tools", "run.py", "training") + [ROOT / "README.md"]
    files += sorted((ROOT / "docs").glob("*.md")) + sorted((ROOT / "docs").glob("*/journal.md"))
    for path in files:
        text = path.read_text()
        for tok in tokens:
            assert not re.search(r"\b" + re.escape(tok) + r"\b", text), (path, tok)


def test_public_gt_only_in_pipeline_check():
    pat = re.compile(r"possession_gt|evaluation/gt|\"gt\"")
    for path in py_files("mtah", "analysis", "training", "run.py"):
        assert not pat.search(path.read_text()), path


def test_repo_data_free():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout
    assert not re.search(r"\.(mp4|csv|npy|parquet|wav)$|BAR-ATH|cache/|tactical/", out, re.M)
