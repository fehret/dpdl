from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DPDL_DIR = REPO_ROOT / "dpdl"
LAUNCHER = REPO_ROOT / "scripts" / "REPLICATE-BISR-STUDY-CIFAR10.sh"


def _run_launcher(*, methods: str, regimes: str, include_controls: str = "0") -> str:
    env = os.environ.copy()
    env.update(
        {
            "SUBMIT_MODE": "print",
            "TRIALS": "1",
            "METHODS": methods,
            "REGIMES": regimes,
            "INCLUDE_CONTROLS": include_controls,
        }
    )
    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=DPDL_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def test_amplified_gaussian_launcher_row_uses_bnb_balls_in_bins_contract() -> None:
    stdout = _run_launcher(methods="dpsgd", regimes="amplified")

    assert "Submitting amplified/dpsgd" in stdout
    assert "--noise-mechanism gaussian" in stdout
    assert "--accountant bnb" in stdout
    assert "--sampling-mode balls_in_bins" in stdout
    assert "--bnb-b 98" in stdout
    assert "--accountant bsr" not in stdout


def test_bandinvmf_rows_emit_live_commands() -> None:
    stdout = _run_launcher(methods="bandinvmf", regimes="amplified nonamplified")

    assert "Submitting amplified/bandinvmf" in stdout
    assert "Submitting nonamplified/bandinvmf" in stdout
    assert "--noise-mechanism bandinvmf" in stdout
    assert "--accountant bnb" in stdout
    assert "--accountant bsr" in stdout
