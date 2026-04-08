from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DPDL_DIR = REPO_ROOT / "dpdl"
LAUNCHER = REPO_ROOT / "scripts" / "REPLICATE-BISR-PAPER-CIFAR10-EXACT.sh"


def _run_launcher(*, methods: str, regimes: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "SUBMIT_MODE": "print",
            "TRIALS": "1",
            "METHODS": methods,
            "REGIMES": regimes,
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=DPDL_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_exact_launcher_preflight_prints_shared_contract_and_approximate_note() -> None:
    proc = _run_launcher(methods="dpsgd", regimes="amplified")

    assert "Resolved shared contract: model=bsr-test-net optimizer=paper-sgd" in proc.stderr
    assert "epochs=10" in proc.stderr
    assert "Approximate row amplified/dpsgd (DP-SGD)" in proc.stderr
    assert "ceil(50000 / 512) * 10 = 98 * 10 = 980" in proc.stderr


def test_exact_launcher_nonamplified_bsr_uses_bsr_z_std_and_epochs() -> None:
    proc = _run_launcher(methods="bsr", regimes="non-amplified")

    assert "Submitting non-amplified/bsr (BSR) status=exact" in proc.stdout
    assert "--bsr-z-std 3.3" in proc.stdout
    assert "--epochs 10" in proc.stdout
    assert "--noise-multiplier" not in proc.stdout


def test_exact_launcher_bandinvmf_rows_map_directly_to_bandinvmf() -> None:
    proc = _run_launcher(methods="bandinvmf", regimes="amplified non-amplified")

    assert "Submitting amplified/bandinvmf (Band-Inv-MF) status=exact" in proc.stdout
    assert "Submitting non-amplified/bandinvmf (Band-Inv-MF) status=exact" in proc.stdout
    assert "--noise-mechanism bandinvmf" in proc.stdout
    assert "--accountant bnb" in proc.stdout
    assert "--accountant bsr" in proc.stdout


def test_exact_launcher_experiment_name_encodes_row_identity() -> None:
    proc = _run_launcher(methods="dpsgd", regimes="amplified")

    assert "CIFAR10-BISR-PAPER-EXACT-amplified-DP-SGD-lr0.1-bw1-trial0-approximate" in proc.stdout
