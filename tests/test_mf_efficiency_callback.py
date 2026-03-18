import json
import math
from types import SimpleNamespace

import torch

from dpdl.callbacks.mf_efficiency import MFEfficiencyMetricsCallback


def _trainer(
    *,
    total_steps: int,
    mechanism: str,
    sampling_mode: str | None,
    mechanism_state: dict,
    sigma: float | None,
    trial_index: int | None = None,
):
    optimizer = SimpleNamespace()
    if sigma is not None:
        optimizer.noise_multiplier = float(sigma)
    privacy_engine = SimpleNamespace(
        noise_mechanism_config=SimpleNamespace(mechanism_state=dict(mechanism_state)),
        sampling_semantics=SimpleNamespace(sampling_mode=sampling_mode),
    )
    return SimpleNamespace(
        total_steps=int(total_steps),
        noise_mechanism=mechanism,
        sampling_mode=sampling_mode,
        optimizer=optimizer,
        privacy_engine=privacy_engine,
        trial_index=trial_index,
    )


def test_mf_efficiency_callback_gaussian_identity_known_value(tmp_path):
    callback = MFEfficiencyMetricsCallback(log_dir=tmp_path)
    callback._is_global_zero = lambda: True
    trainer = _trainer(
        total_steps=4,
        mechanism="gaussian",
        sampling_mode="poisson",
        mechanism_state={},
        sigma=2.0,
        trial_index=7,
    )

    callback.on_train_end(trainer)
    summary = callback.latest_summary
    assert summary is not None
    assert summary["mf_efficiency_status"] == "computed"
    # For n=4 and C=I, ||A||_F^2 = 1+2+3+4 = 10, sigma^2 = 4 => 40.
    assert summary["mse_prefix"] == 40.0
    assert summary["rmse_prefix"] == math.sqrt(40.0)
    assert summary["trial_index"] == 7

    persisted = json.loads((tmp_path / "mf_efficiency.json").read_text())
    assert persisted["mse_prefix"] == 40.0
    assert persisted["rmse_prefix"] == math.sqrt(40.0)


def test_mf_efficiency_callback_unavailable_without_sigma(tmp_path):
    callback = MFEfficiencyMetricsCallback(log_dir=tmp_path)
    callback._is_global_zero = lambda: True
    trainer = _trainer(
        total_steps=4,
        mechanism="gaussian",
        sampling_mode="poisson",
        mechanism_state={},
        sigma=None,
    )

    callback.on_train_end(trainer)
    summary = callback.latest_summary
    assert summary is not None
    assert summary["mf_efficiency_status"] == "unavailable"
    assert summary["mf_efficiency_reason"] == "missing_or_invalid_sigma"
    assert summary["mse_prefix"] is None
    assert summary["rmse_prefix"] is None


def test_mf_efficiency_callback_toeplitz_identity_like_and_json_stability(tmp_path):
    callback = MFEfficiencyMetricsCallback(log_dir=tmp_path)
    callback._is_global_zero = lambda: True
    trainer = _trainer(
        total_steps=3,
        mechanism="bsr",
        sampling_mode="torch_sampler",
        mechanism_state={"coeffs": [1.0, 0.0], "bsr_bands": 2},
        sigma=1.0,
        trial_index=3,
    )

    callback.on_train_end(trainer)
    summary = callback.latest_summary
    assert summary is not None
    assert summary["mf_efficiency_status"] == "computed"
    # For n=3 and C=I, ||A||_F^2 = 1+2+3 = 6.
    assert summary["mse_prefix"] == 6.0
    assert summary["rmse_prefix"] == math.sqrt(6.0)

    lines = (tmp_path / "mf_efficiency_trials.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert isinstance(row["mse_prefix"], float)
    assert isinstance(row["rmse_prefix"], float)


def test_mf_efficiency_callback_uses_mechanism_z_std_when_optimizer_sigma_missing(tmp_path):
    callback = MFEfficiencyMetricsCallback(log_dir=tmp_path)
    callback._is_global_zero = lambda: True
    trainer = _trainer(
        total_steps=3,
        mechanism="bsr",
        sampling_mode="torch_sampler",
        mechanism_state={"coeffs": [1.0, 0.0], "z_std": 0.5},
        sigma=None,
    )

    callback.on_train_end(trainer)
    summary = callback.latest_summary
    assert summary is not None
    assert summary["mf_efficiency_status"] == "computed"
    assert summary["sigma_source"] == "mechanism_state.z_std"
    assert summary["sigma_c"] == 0.5
    # For n=3 and C=I, ||A||_F^2 = 6 and sigma^2 = 0.25.
    assert summary["mse_prefix"] == 1.5
    assert summary["rmse_prefix"] == math.sqrt(1.5)


def test_mf_efficiency_callback_bnb_uses_c_matrix_horizon_when_aligned(tmp_path):
    callback = MFEfficiencyMetricsCallback(log_dir=tmp_path)
    callback._is_global_zero = lambda: True
    trainer = _trainer(
        total_steps=5,  # training horizon
        mechanism="gaussian",
        sampling_mode="balls_in_bins",
        mechanism_state={"bnb_c_matrix": torch.eye(6).tolist(), "bnb_bands": 3},
        sigma=1.0,
    )

    callback.on_train_end(trainer)
    summary = callback.latest_summary
    assert summary is not None
    assert summary["mf_efficiency_status"] == "computed"
    assert summary["horizon_training"] == 5
    assert summary["horizon"] == 5
    # For n=5 and C=I, ||A||_F^2 = 1+2+...+5 = 15.
    assert summary["mse_prefix"] == 15.0
    assert summary["rmse_prefix"] == math.sqrt(15.0)


def test_mf_efficiency_callback_invalid_z_std_does_not_fallback_to_optimizer(tmp_path):
    callback = MFEfficiencyMetricsCallback(log_dir=tmp_path)
    callback._is_global_zero = lambda: True
    trainer = _trainer(
        total_steps=3,
        mechanism="bsr",
        sampling_mode="torch_sampler",
        mechanism_state={"coeffs": [1.0, 0.0], "z_std": float("nan")},
        sigma=1.0,
    )

    callback.on_train_end(trainer)
    summary = callback.latest_summary
    assert summary is not None
    assert summary["mf_efficiency_status"] == "unavailable"
    assert summary["sigma_source"] == "mechanism_state.z_std_invalid"
    assert summary["mf_efficiency_reason"] == "missing_or_invalid_sigma"


def test_mf_efficiency_callback_invalid_optimizer_sigma_is_handled_gracefully(tmp_path):
    callback = MFEfficiencyMetricsCallback(log_dir=tmp_path)
    callback._is_global_zero = lambda: True
    trainer = _trainer(
        total_steps=4,
        mechanism="gaussian",
        sampling_mode="poisson",
        mechanism_state={},
        sigma=None,
    )
    trainer.optimizer.noise_multiplier = "not-a-number"

    callback.on_train_end(trainer)
    summary = callback.latest_summary
    assert summary is not None
    assert summary["mf_efficiency_status"] == "unavailable"
    assert summary["mf_efficiency_reason"] == "missing_or_invalid_sigma"
