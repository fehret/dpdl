from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DPDL_DIR = REPO_ROOT / "dpdl"
LAUNCHER = REPO_ROOT / "scripts" / "OPTIMIZE-BISR-STUDY-CIFAR10.sh"


def _run_launcher(*, methods: str, regimes: str, extra_env: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    env.update(
        {
            "SUBMIT_MODE": "print",
            "METHODS": methods,
            "REGIMES": regimes,
            "SEED_START": "42",
            "SEED_END": "42",
            "N_TRIALS": "3",
            "LOG_DIR_BASE": "outputs/test-optimize-bisr-study-cifar10",
        }
    )
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=DPDL_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def test_optimize_launcher_targets_learning_rate_only_for_amplified_bisr() -> None:
    stdout = _run_launcher(methods="bisr", regimes="amplified")

    assert "Submitting amplified/bisr" in stdout
    assert "run.py optimize" in stdout
    assert "--target-hypers learning_rate" in stdout
    assert "--target-hypers max_grad_norm" not in stdout
    assert "--noise-mechanism bisr" in stdout
    assert "--accountant bnb" in stdout
    assert "--sampling-mode balls_in_bins" in stdout
    assert "--noise-multiplier 4.36940202739" in stdout
    assert "--target-epsilon" not in stdout
    assert "--bsr-bands 4" in stdout
    assert "--max-grad-norm 10.0" in stdout


def test_optimize_launcher_uses_fixed_batch_identity_control_for_nonamplified_dpsgd() -> None:
    stdout = _run_launcher(methods="dpsgd", regimes="nonamplified")

    assert "Submitting nonamplified/dpsgd" in stdout
    assert "--noise-mechanism bsr" in stdout
    assert "--accountant bsr" in stdout
    assert "--sampling-mode torch_sampler" in stdout
    assert "--bsr-bands 1" in stdout
    assert "--bsr-coeffs 1.0" in stdout
    assert "--target-hypers learning_rate" in stdout


def test_optimize_launcher_forwards_bnb_sampling_controls() -> None:
    stdout = _run_launcher(
        methods="bsr",
        regimes="amplified",
        extra_env={
            "BNB_CALIBRATION_MODE": "optimistic",
            "BNB_NUM_SAMPLES": "200000",
        },
    )

    assert "--bnb-calibration-mode optimistic" in stdout
    assert "--bnb-num-samples 200000" in stdout
    assert "--optuna-config conf/optuna_hypers_bisr_study_cifar10_lr_only.conf" in stdout


def test_optimize_launcher_allows_noise_multiplier_override() -> None:
    stdout = _run_launcher(
        methods="bisr",
        regimes="amplified",
        extra_env={"NOISE_MULTIPLIER_OVERRIDE": "7.77"},
    )

    assert "--noise-multiplier 7.77" in stdout
    assert "row_sigma=4.36940202739" in stdout
