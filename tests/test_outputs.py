"""The check scripts print PASS / FAIL lines only. Needs the fixture of analysis/clean/fixture.py:

    MTAH_FIXTURE=<dir> MTAH_FIXTURE_REG=<registry json> MTAH_FIXTURE_WEIGHTS=<dir> pytest tests/test_outputs.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = os.environ.get("MTAH_FIXTURE")
need = pytest.mark.skipif(not FIX, reason="no fixture")
LINE = re.compile(r"^[AIK]\d+[a-z]? (PASS|FAIL)$")


def only_verdicts(cmd, env):
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    lines = out.strip().splitlines()
    assert lines and all(LINE.match(x) for x in lines), lines


@need
def test_check_inputs():
    env = dict(os.environ, MTAH_DATA_DIR=FIX, PYTHONPATH=str(ROOT))
    blocks = Path(FIX, "blocks.json").read_text()
    only_verdicts([sys.executable, "-m", "analysis.clean.check_inputs", "analysis/clean/m4_quantiles.json",
                   "--blocks", blocks], env)


@need
def test_check_output():
    env = dict(os.environ, MTAH_DATA_DIR=FIX, PYTHONPATH=str(ROOT))
    blocks = Path(FIX, "blocks.json").read_text()
    only_verdicts([sys.executable, "-m", "analysis.clean.check_output", str(Path(FIX, "out.csv")),
                   os.environ["MTAH_FIXTURE_REG"], "--a6", "--blocks", blocks, "--weights",
                   os.environ["MTAH_FIXTURE_WEIGHTS"]], env)


@need
def test_pipeline_check_on_a_synthetic_file():
    env = dict(os.environ)
    only_verdicts([sys.executable, "tools/pipeline_check.py", str(Path(FIX, "out.csv")),
                   "--gt", str(Path(FIX, "gt_synth.csv"))], env)


@need
def test_run_py_loads_no_training_stack():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["MTAH_DATA_DIR"] = FIX
    code = ("import sys, runpy; sys.argv = ['run.py', '--blocks', {!r}, '--weights', {!r}, '--out', {!r}];\n"
            "try:\n    runpy.run_path('run.py', run_name='__main__')\nexcept SystemExit:\n    pass\n"
            "assert not {{'sklearn', 'scipy', 'torch'}} & set(sys.modules), 'training stack loaded'\n").format(
        Path(FIX, "blocks.json").read_text(), os.environ["MTAH_FIXTURE_WEIGHTS"], str(Path(FIX, "out_m.csv")))
    subprocess.run([str(ROOT / ".venv-run" / "bin" / "python"), "-c", code], cwd=ROOT, env=env, check=True)
