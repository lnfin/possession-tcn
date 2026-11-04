"""The harness of section 0.4: matches, units, references, official metric, summary W.

The scorer is loaded directly from the official repository; no logic of it is copied. The periods
are passed explicitly: pm.valid_runs would glue halves that follow each other without a gap.
"""

import hashlib
import importlib.util
import sys

import numpy as np

from mtah import paths
from mtah.external import idsse_open, metrica_open, reference

METRICA = ("metrica1", "metrica2", "metrica3")
IDSSE = idsse_open.MATCHES
GAMES_TRAIN = METRICA + IDSSE                # m = index here, 0-9
GAMES_EVAL = METRICA + IDSSE[:-1]            # J03WR9: goalkeeper track defect
FORTUNA_EVAL = ("J03WOH", "J03WOY", "J03WPY", "J03WQQ")
UNITS = {"metrica1": ("metrica1",), "metrica2": ("metrica2",), "metrica3": ("metrica3",),
         "J03WMX": ("J03WMX",), "J03WN1": ("J03WN1",), "Fortuna": FORTUNA_EVAL}
UNIT_HOLD = {"metrica1": ("metrica1",), "metrica2": ("metrica2",), "metrica3": ("metrica3",),
             "J03WMX": ("J03WMX",), "J03WN1": ("J03WN1",), "Fortuna": FORTUNA_EVAL + ("J03WR9",)}
UNIT_W = {"metrica1": 1 / 6, "metrica2": 1 / 6, "metrica3": 1 / 6, "J03WMX": 1 / 12, "J03WN1": 1 / 12,
          "Fortuna": 1 / 3}
UNIT_NAMES = tuple(UNITS)
SOURCE_UNITS = {"M": METRICA, "I": ("J03WMX", "J03WN1", "Fortuna")}
REFS = ("R", "R25", "R75", "Rdef")           # E_def, equal weights: decision 8
OFFICIAL_MD5 = "7718cb8a237452b48b35713984d31c75"
_pm = None


def load_pm():
    """The official possession_metrics.py, md5 checked, by importlib."""
    global _pm
    if _pm is None:
        path = paths.official_repo() / "evaluation" / "possession_metrics.py"
        got = hashlib.md5(path.read_bytes()).hexdigest()
        if got != OFFICIAL_MD5:
            raise RuntimeError("{}: md5 {}".format(path, got))
        spec = importlib.util.spec_from_file_location("possession_metrics", path)
        _pm = importlib.util.module_from_spec(spec)
        sys.modules["possession_metrics"] = _pm
        spec.loader.exec_module(_pm)
    return _pm


def runs_of(periods):
    return [(a, b + 1) for _, a, b in periods]


def spell_macro(g, p, periods):
    """(spell, macro) of chars p against chars g, by the formula of section 0.4."""
    pm = load_pm()
    gc, pc = pm.label_codes(g), pm.label_codes(p)
    runs = runs_of(periods)
    spell = pm.spell_scores(gc, pm.fill_abstentions(pc, runs), runs)["mean_f1"]
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gc, pc)))
    return float(spell), float(macro)


def W(x):
    """Summary of a {unit: value} mapping: sum of w_u x_u."""
    return float(sum(UNIT_W[u] * x[u] for u in UNIT_NAMES))


def source_mean(x, source, units=UNIT_NAMES):
    """Mean by source over the given units: sum w_u x_u / sum w_u."""
    keep = [u for u in SOURCE_UNITS[source] if u in units]
    den = sum(UNIT_W[u] for u in keep)
    return float(sum(UNIT_W[u] * x[u] for u in keep) / den)


def load_game(name):
    return metrica_open.load(name) if name in METRICA else idsse_open.load(name)


def references(name, game, ref_params):
    """{R, R25, R75, Rdef: int8 codes}, stoppage log, info of one match; R of table 1.8 or the flags."""
    per = game["periods"]
    info = {}
    if name in METRICA:
        y, log = reference.labels(game, ref_params, return_log=True)
        ydef = reference.metrica_def(game, y)
    else:
        y, log = game["label"].copy(), []
        ydef, lag, share = reference.idsse_def(idsse_open.events(name), y, per)
        info = {"lag": lag, "share": share}
    return {"R": y, "R25": reference.merge(y, per, 25), "R75": reference.merge(y, per, 75),
            "Rdef": ydef}, log, info
