from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DPDL_DIR = REPO_ROOT / "dpdl"
LAUNCHER = REPO_ROOT / "scripts" / "OPTIMIZE-BISR-STUDY-CIFAR10-LR-CLIP.sh"


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
            "LOG_DIR_BASE": "outputs/test-optimize-bisr-study-cifar10-lr-clip",
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


def test_optimize_lr_clip_launcher_targets_learning_rate_and_clip_for_amplified_bisr() -> None:
    stdout = _run_launcher(methods="bisr", regimes="amplified")

    assert "Submitting amplified/bisr" in stdout
    assert "run.py optimize" in stdout
    assert "--target-hypers learning_rate" in stdout
    assert "--target-hypers max_grad_norm" in stdout
    assert "--optuna-config conf/optuna_hypers_bisr_study_cifar10_lr_clip.conf" in stdout
    assert "--noise-mechanism bisr" in stdout
    assert "--accountant bnb" in stdout
    assert "--sampling-mode balls_in_bins" in stdout
    assert "--noise-multiplier 4.36940202739" in stdout
    assert "--max-grad-norm 10.0" in stdout


def test_optimize_lr_clip_launcher_supports_full_comparison_table_rows() -> None:
    stdout = _run_launcher(methods="dpsgd bsr bisr bandmf bandinvmf", regimes="nonamplified")

    assert "Submitting nonamplified/dpsgd" in stdout
    assert "Submitting nonamplified/bandmf" in stdout
    assert "Submitting nonamplified/bisr" in stdout
    assert "Submitting nonamplified/bandinvmf" in stdout
    assert "Submitting nonamplified/bsr" in stdout
    assert "--noise-mechanism bandmf" in stdout
    assert "--noise-mechanism bandinvmf" in stdout
    assert "--noise-mechanism bisr" in stdout
    assert "--noise-mechanism bsr" in stdout
    assert "--bsr-coeffs 1.0" in stdout


def test_optimize_lr_clip_launcher_defaults_to_full_method_set() -> None:
    stdout = _run_launcher(methods="dpsgd bsr bisr bandmf bandinvmf", regimes="amplified nonamplified")

    assert "METHODS=dpsgd bsr bisr bandmf bandinvmf" in stdout
    assert "REGIMES=amplified nonamplified" in stdout


def test_optimize_lr_clip_launcher_allows_noise_multiplier_override() -> None:
    stdout = _run_launcher(
        methods="bandinvmf",
        regimes="amplified",
        extra_env={"NOISE_MULTIPLIER_OVERRIDE": "7.77"},
    )

    assert "--noise-multiplier 7.77" in stdout
    assert "row_sigma=7.74664888071" in stdout
