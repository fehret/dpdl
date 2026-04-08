from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from dpdl.pretrained_benchmark_manifest import iter_rows
from dpdl.pretrained_benchmark_sigma_calibration import CalibratedSigmaRow, build_report_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
DPDL_DIR = REPO_ROOT / "dpdl"
LAUNCHER = REPO_ROOT / "scripts" / "OPTIMIZE-PRETRAINED-BENCHMARK-LR-ONLY.sh"


def _write_sigma_report(tmp_path: Path) -> Path:
    calibrated_rows = []
    for row in iter_rows():
        row_dict = {
            "row_id": row.row_id,
            "dataset_name": row.dataset_name,
            "label_field": row.label_field,
            "dataset_size": row.dataset_size,
            "epochs": row.epochs,
            "steps_per_epoch": row.steps_per_epoch,
            "total_steps": row.total_steps,
            "regime": row.regime,
            "method": row.method,
            "epsilon": row.epsilon,
            "delta": row.delta,
            "model_name": row.model_name,
            "optimizer": row.optimizer,
            "pretrained": row.pretrained,
            "batch_size": row.batch_size,
            "physical_batch_size": row.physical_batch_size,
            "max_grad_norm": row.max_grad_norm,
            "bands": row.bands,
            "noise_mechanism": row.noise_mechanism,
            "accountant": row.accountant,
            "sampling_mode": row.sampling_mode,
            "poisson_sampling": row.poisson_sampling,
            "explicit_coeffs": row.explicit_coeffs,
            "calibrated_for": row.calibrated_for,
            "noise_multiplier": 1.2345,
            "bnb_num_samples": 100000,
            "bnb_calibration_mode": "optimistic",
        }
        calibrated_rows.append(CalibratedSigmaRow(**row_dict))
    report = build_report_payload(calibrated_rows)
    path = tmp_path / "sigma_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _run_launcher(
    tmp_path: Path,
    *,
    datasets: str,
    methods: str,
    regimes: str,
    epsilons: str,
    extra_env: dict[str, str] | None = None,
) -> str:
    env = os.environ.copy()
    env.update(
        {
            "SUBMIT_MODE": "print",
            "DATASETS": datasets,
            "METHODS": methods,
            "REGIMES": regimes,
            "EPSILONS": epsilons,
            "SEED_START": "42",
            "SEED_END": "42",
            "N_TRIALS": "3",
            "LOG_DIR_BASE": str(tmp_path / "logs"),
            "SIGMA_REPORT": str(_write_sigma_report(tmp_path)),
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


def test_pretrained_optimize_launcher_covers_expected_matrix_contract(tmp_path: Path) -> None:
    stdout = _run_launcher(
        tmp_path,
        datasets="uoft-cs/cifar100 dpdl-benchmark/sun397 dpdl-benchmark/cassava",
        methods="dpsgd idb1 bsr bisr bandmf bandinvmf",
        regimes="amplified nonamplified",
        epsilons="0.5 1 2 4 8",
    )

    assert "DATASETS=uoft-cs/cifar100 dpdl-benchmark/sun397 dpdl-benchmark/cassava" in stdout
    assert "METHODS=dpsgd idb1 bsr bisr bandmf bandinvmf" in stdout
    assert "REGIMES=amplified nonamplified" in stdout
    assert "EPSILONS=0.5 1 2 4 8" in stdout
    assert "OPTIMIZER=paper-sgd" in stdout
    assert "BATCH_SIZE=512 PHYSICAL_BATCH_SIZE=32" in stdout
    assert "MAX_GRAD_NORM=10.0 BANDS=4" in stdout
    assert "PRETRAINED=true" in stdout
    assert "SIGMA_SOURCE=report_json" in stdout
    assert "SKIP_POLICY=runtime_file_or_squeue" in stdout
    assert "TRUE_DPSGD=gaussian+prv+poisson+target_epsilon" in stdout


def test_pretrained_optimize_launcher_uses_dataset_specific_metadata_and_fixed_sigma(tmp_path: Path) -> None:
    stdout = _run_launcher(
        tmp_path,
        datasets="uoft-cs/cifar100",
        methods="bisr",
        regimes="amplified",
        epsilons="1",
    )

    assert "Submitting dataset=uoft-cs/cifar100 regime=amplified method=bisr epsilon=1" in stdout
    assert "--dataset-label-field fine_label" in stdout
    assert "--total-steps 784" in stdout
    assert "--noise-multiplier 1.2345" in stdout
    assert "--target-epsilon" not in stdout
    assert "--pretrained" in stdout
    assert "--no-pretrained" not in stdout
    assert "--target-hypers learning_rate" in stdout
    assert "--optuna-config conf/optuna_hypers_pretrained_benchmark_lr_only.conf" in stdout
    assert "--sampling-mode balls_in_bins" in stdout
    assert "--bnb-b 98" in stdout


def test_pretrained_optimize_launcher_handles_nonamplified_identity_control_and_split_policy_note(tmp_path: Path) -> None:
    stdout = _run_launcher(
        tmp_path,
        datasets="dpdl-benchmark/cassava",
        methods="idb1",
        regimes="nonamplified",
        epsilons="4",
    )

    assert "BO_SPLIT_POLICY=heldout_split_hpo_then_train_plus_valid_final" in stdout
    assert "SIGMA_CONTRACT=final_evaluation_round" in stdout
    assert "--dataset-label-field label" in stdout
    assert "--total-steps 384" in stdout
    assert "--noise-multiplier 1.2345" in stdout
    assert "--noise-mechanism bsr" in stdout
    assert "--accountant bsr" in stdout
    assert "--sampling-mode torch_sampler" in stdout
    assert "--bsr-bands 1" in stdout
    assert "--bsr-coeffs 1.0" in stdout


def test_pretrained_optimize_launcher_emits_true_dpsgd_poisson_prv_row(tmp_path: Path) -> None:
    stdout = _run_launcher(
        tmp_path,
        datasets="uoft-cs/cifar100",
        methods="dpsgd",
        regimes="nonamplified",
        epsilons="1",
    )

    assert "Submitting dataset=uoft-cs/cifar100 regime=poissonprv method=dpsgd epsilon=1" in stdout
    assert "--noise-mechanism gaussian" in stdout
    assert "--accountant prv" in stdout
    assert "--poisson-sampling" in stdout
    assert "--target-epsilon 1" in stdout
    assert "--noise-multiplier" not in stdout
    assert "--bsr-coeffs" not in stdout
    assert "--bsr-bands" not in stdout


def test_pretrained_optimize_launcher_uses_new_lr_range_config(tmp_path: Path) -> None:
    stdout = _run_launcher(
        tmp_path,
        datasets="dpdl-benchmark/sun397",
        methods="bandinvmf",
        regimes="nonamplified",
        epsilons="8",
    )

    assert "--optuna-config conf/optuna_hypers_pretrained_benchmark_lr_only.conf" in stdout
    assert "--n-trials 3" in stdout
