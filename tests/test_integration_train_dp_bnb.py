from pathlib import Path
import os
import sys
import copy

import pytest
import torch
import opacus
import datasets

pytest.importorskip('torch')
pytest.importorskip('opacus')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.configurationmanager import ConfigurationManager
from dpdl.trainer import TrainerFactory
import dpdl.trainer as trainer_mod

from integration_utils import (
    assert_config_and_hyperparams,
    assert_runtime,
    assert_test_metrics,
    base_env,
    run_distributed,
)


def _capture_dp_handoff(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    command: str = "train",
    noise_mechanism: str,
    use_explicit_coeffs: bool,
    total_steps: int,
    sampling_mode: str = "balls_in_bins",
    bnb_b: int | None = 2,
    bnb_p: float | None = None,
    blt_buffers: int | None = None,
    world_size: int = 1,
    batch_size: int = 4,
) -> tuple[dict, dict, object, int]:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(
        trainer_mod.opacus.distributed,
        "DifferentiallyPrivateDistributedDataParallel",
        lambda model: model,
    )

    captured: dict[str, dict] = {}
    orig_make_private = opacus.PrivacyEngine.make_private

    def wrapped_make_private(self, *args, **kwargs):
        captured["make_private_kwargs"] = copy.deepcopy(kwargs)
        captured["dataset_size"] = int(len(kwargs["data_loader"].dataset))
        captured["pre_state"] = copy.deepcopy(kwargs["noise_mechanism_config"].mechanism_state)
        captured["sampling_semantics"] = copy.deepcopy(kwargs.get("sampling_semantics"))
        result = orig_make_private(self, *args, **kwargs)
        captured["post_state"] = copy.deepcopy(self.noise_mechanism_config.mechanism_state)
        return result

    monkeypatch.setattr(opacus.PrivacyEngine, "make_private", wrapped_make_private)

    cli_params = {
        "command": command,
        "device": "cpu",
        "dataset_name": "local-image",
        "dataset_path": str(image_dataset_path),
        "model_name": "vit_tiny_patch16_224.augreg_in21k",
        "privacy": True,
        "use_steps": True,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "physical_batch_size": batch_size,
        "num_workers": 0,
        "seed": 42,
        "split_seed": 42,
        "max_grad_norm": 1.0,
        "poisson_sampling": False,
        "target_epsilon": None,
        "noise_batch_ratio": None,
        "noise_mechanism": noise_mechanism,
        "accountant": "bnb",
        "sampling_mode": sampling_mode,
        "noise_multiplier": 10.0,
        "pretrained": False,
    }
    if noise_mechanism in ("bsr", "bisr"):
        cli_params["bsr_bands"] = 2
    if noise_mechanism == "bifr":
        cli_params["bsr_bands"] = 2
        cli_params["bifr_frac"] = 0.25
    if noise_mechanism == "blt" and blt_buffers is not None:
        cli_params["blt_buffers"] = blt_buffers
    if bnb_b is not None:
        cli_params["bnb_b"] = bnb_b
    if bnb_p is not None:
        cli_params["bnb_p"] = bnb_p
    if use_explicit_coeffs:
        cli_params["bsr_coeffs"] = [1.0, 0.2] if noise_mechanism == "bsr" else [1.0, -0.5]

    cfg_mgr = ConfigurationManager(cli_params)
    TrainerFactory.get_trainer(cfg_mgr)

    return (
        captured["pre_state"],
        captured["post_state"],
        captured["sampling_semantics"],
        captured["dataset_size"],
    )


def _build_train_test_only_image_dataset(src: Path, dst: Path) -> Path:
    ds = datasets.load_from_disk(str(src))
    reduced = datasets.DatasetDict(
        {
            "train": ds["train"],
            "test": ds["test"],
        }
    )
    reduced.save_to_disk(str(dst))
    return dst


def test_capture_dp_handoff_marks_bnb_chunk_shard_for_distributed_runtime(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(
        trainer_mod.opacus.distributed,
        "DifferentiallyPrivateDistributedDataParallel",
        lambda model: model,
    )

    captured: dict[str, dict] = {}
    orig_make_private = opacus.PrivacyEngine.make_private

    def wrapped_make_private(self, *args, **kwargs):
        captured["kwargs"] = copy.deepcopy(kwargs)
        return orig_make_private(self, *args, **kwargs)

    monkeypatch.setattr(opacus.PrivacyEngine, "make_private", wrapped_make_private)

    cli_params = {
        "command": "train",
        "device": "cpu",
        "dataset_name": "local-image",
        "dataset_path": str(image_dataset_path),
        "model_name": "vit_tiny_patch16_224.augreg_in21k",
        "privacy": True,
        "use_steps": True,
        "total_steps": 2,
        "batch_size": 4,
        "physical_batch_size": 4,
        "num_workers": 0,
        "seed": 42,
        "split_seed": 42,
        "max_grad_norm": 1.0,
        "poisson_sampling": False,
        "target_epsilon": None,
        "noise_batch_ratio": None,
        "noise_mechanism": "bsr",
        "accountant": "bnb",
        "sampling_mode": "balls_in_bins",
        "bnb_b": 2,
        "noise_multiplier": 10.0,
        "bsr_bands": 2,
        "pretrained": False,
    }

    cfg_mgr = ConfigurationManager(cli_params)
    TrainerFactory.get_trainer(cfg_mgr)

    kwargs = captured["kwargs"]
    assert kwargs["bnb_distributed_dp_runtime"] is True
    assert kwargs["bnb_distributed_mode"] == "chunk_shard"


def test_capture_dp_handoff_allows_blt_in_distributed_runtime(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre_state, post_state, sampling_semantics, dataset_size = _capture_dp_handoff(
        image_dataset_path,
        monkeypatch,
        noise_mechanism="blt",
        use_explicit_coeffs=False,
        total_steps=2,
        bnb_b=2,
        blt_buffers=4,
        world_size=8,
    )

    assert sampling_semantics.sampling_mode == "balls_in_bins"
    assert sampling_semantics.privacy_metadata["bins"] == 2
    assert sampling_semantics.privacy_metadata["bands"] == 2
    assert dataset_size > 0
    assert pre_state["blt_buffers"] == 4
    assert post_state["blt_buffers"] == 4


def test_optimize_balls_in_bins_mf_uses_full_train_split_as_workload_reference(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _build_train_test_only_image_dataset(
        image_dataset_path,
        tmp_path / "train-test-only-image",
    )
    pre_state, post_state, sampling_semantics, dataset_size = _capture_dp_handoff(
        dataset_path,
        monkeypatch,
        command="optimize",
        noise_mechanism="bsr",
        use_explicit_coeffs=False,
        total_steps=2,
        batch_size=3,
        bnb_b=None,
    )

    assert dataset_size == 18
    assert sampling_semantics.sampling_mode == "balls_in_bins"
    assert sampling_semantics.privacy_metadata["bins"] == 7
    assert pre_state["bnb_bins"] == 7
    assert post_state["bnb_bins"] == 7


def _run_dp_bnb(
    tmp_path: Path,
    image_dataset_path: Path,
    *,
    experiment: str,
    use_target_epsilon: bool,
    total_steps: int = 1,
    sampling_mode: str = 'balls_in_bins',
    noise_mechanism: str = 'gaussian',
    accountant: str = 'bnb',
    model_name: str = 'vit_tiny_patch16_224.augreg_in21k',
    model_args: list[str] | None = None,
    use_explicit_coeffs: bool = True,
    blt_buffers: int | None = None,
) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    env = base_env()

    cmd_args = [
        'run.py',
        'train',
        '--device',
        'cpu',
        '--dataset-name',
        'local-image',
        '--dataset-path',
        str(image_dataset_path),
        '--model-name',
        model_name,
        '--privacy',
        '--use-steps',
        '--total-steps',
        str(total_steps),
        '--batch-size',
        '4',
        '--physical-batch-size',
        '4',
        '--num-workers',
        '0',
        '--seed',
        '42',
        '--split-seed',
        '42',
        '--max-grad-norm',
        '1.0',
        '--no-poisson-sampling',
        '--noise-mechanism',
        noise_mechanism,
        '--accountant',
        accountant,
        '--sampling-mode',
        sampling_mode,
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        experiment,
    ]
    if sampling_mode in ('balls_in_bins', 'b_min_sep'):
        cmd_args.extend(['--bnb-b', '2'])

    if noise_mechanism == 'bsr':
        cmd_args.extend(['--bsr-bands', '2'])
        if use_explicit_coeffs:
            cmd_args.extend(['--bsr-coeffs', '1.0', '--bsr-coeffs', '0.2'])
    elif noise_mechanism == 'bisr':
        cmd_args.extend(['--bsr-bands', '2'])
        if use_explicit_coeffs:
            cmd_args.extend(['--bsr-coeffs', '1.0', '--bsr-coeffs', '-0.5'])
    elif noise_mechanism == 'bifr':
        cmd_args.extend(['--bsr-bands', '2', '--bifr-frac', '0.25'])
    elif noise_mechanism == 'blt':
        if blt_buffers is not None:
            cmd_args.extend(['--blt-buffers', str(blt_buffers)])
    else:
        if noise_mechanism != 'gaussian':
            raise AssertionError(f'unsupported test noise_mechanism {noise_mechanism!r}')

    cmd_args.append('--no-pretrained')

    if model_args:
        cmd_args.extend(model_args)

    if use_target_epsilon:
        cmd_args.extend(['--target-epsilon', '8'])
    else:
        cmd_args.extend(['--noise-multiplier', '10.0'])

    run_distributed(cmd_args, env, repo_root)

    expected_hypers = {
        'epochs': None,
        'total_steps': total_steps,
        'batch_size': 4,
        'max_grad_norm': 1.0,
    }
    if use_target_epsilon:
        expected_hypers['target_epsilon'] = 8.0
    else:
        expected_hypers['target_epsilon'] = None
        expected_hypers['noise_multiplier'] = 10.0

    expected_config = {
        'command': 'train',
        'device': 'cpu',
        'dataset_name': 'local-image',
        'dataset_path': str(image_dataset_path),
        'model_name': model_name,
        'privacy': True,
        'use_steps': True,
        'noise_mechanism': noise_mechanism,
        'accountant': accountant,
        'sampling_mode': sampling_mode,
        'poisson_sampling': False,
    }
    if sampling_mode in ('balls_in_bins', 'b_min_sep'):
        expected_config['bnb_b'] = 2
    if noise_mechanism in ('bsr', 'bisr'):
        expected_config['bsr_bands'] = 2
    if noise_mechanism == 'bifr':
        expected_config['bsr_bands'] = 2
        expected_config['bifr_frac'] = 0.25
    if noise_mechanism == 'blt' and blt_buffers is not None:
        expected_config['blt_buffers'] = blt_buffers
        expected_hypers['blt_buffers'] = blt_buffers

    assert_config_and_hyperparams(
        tmp_path / experiment,
        expected_config=expected_config,
        expected_hyperparams=expected_hypers,
    )

    metrics = assert_test_metrics(tmp_path / experiment)
    assert_runtime(tmp_path / experiment)
    return metrics


@pytest.mark.integration
def test_integration_train_dp_bnb_fixed_noise_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-bnb-fixed-noise',
        use_target_epsilon=False,
    )
    assert 'loss' in metrics


@pytest.mark.integration
def test_integration_train_dp_bnb_fixed_noise_balls_in_bins_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-bnb-fixed-noise-balls-in-bins',
        use_target_epsilon=False,
        sampling_mode='balls_in_bins',
    )
    assert 'loss' in metrics


@pytest.mark.integration
def test_integration_train_dp_bnb_fixed_noise_vgg_reference_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-bnb-fixed-noise-vgg-reference',
        use_target_epsilon=False,
        sampling_mode='balls_in_bins',
        model_name='vgg_bnb_reference',
    )
    assert 'loss' in metrics


@pytest.mark.integration
def test_integration_train_dp_bsr_balls_in_bins_fixed_noise_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-bsr-balls-in-bins-fixed-noise',
        use_target_epsilon=False,
        sampling_mode='balls_in_bins',
        noise_mechanism='bsr',
        accountant='bnb',
    )
    assert 'loss' in metrics


@pytest.mark.integration
def test_integration_train_dp_bsr_balls_in_bins_fixed_noise_autocoeff_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-bsr-balls-in-bins-fixed-noise-autocoeff',
        use_target_epsilon=False,
        total_steps=2,
        sampling_mode='balls_in_bins',
        noise_mechanism='bsr',
        accountant='bnb',
        use_explicit_coeffs=False,
    )
    assert 'loss' in metrics


@pytest.mark.integration
def test_integration_train_dp_bisr_balls_in_bins_fixed_noise_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-bisr-balls-in-bins-fixed-noise',
        use_target_epsilon=False,
        sampling_mode='balls_in_bins',
        noise_mechanism='bisr',
        accountant='bnb',
    )
    assert 'loss' in metrics


@pytest.mark.integration
def test_integration_train_dp_bisr_balls_in_bins_fixed_noise_autocoeff_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-bisr-balls-in-bins-fixed-noise-autocoeff',
        use_target_epsilon=False,
        total_steps=2,
        sampling_mode='balls_in_bins',
        noise_mechanism='bisr',
        accountant='bnb',
        use_explicit_coeffs=False,
    )
    assert 'loss' in metrics


@pytest.mark.integration
def test_integration_train_dp_gaussian_balls_in_bins_fixed_noise_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-gaussian-balls-in-bins-fixed-noise',
        use_target_epsilon=False,
        sampling_mode='balls_in_bins',
        noise_mechanism='gaussian',
        accountant='bnb',
    )
    assert 'loss' in metrics


def test_bsr_balls_in_bins_dpdl_handoff_stays_minimal_and_opacus_resolves(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre_state, post_state, _sampling_semantics, _dataset_size = _capture_dp_handoff(
        image_dataset_path,
        monkeypatch,
        noise_mechanism="bsr",
        use_explicit_coeffs=True,
        total_steps=7,
    )

    for key in (
        "bnb_accountant_coeffs",
        "bnb_accountant_coeffs_source",
        "bnb_c_matrix",
        "bnb_c_matrix_contract",
    ):
        assert key not in pre_state

    assert pre_state["bnb_horizon"] == 7
    assert post_state["bnb_accountant_coeffs_source"] == "raw_c_col"
    assert post_state["bnb_c_matrix"] is not None
    assert post_state["bnb_c_matrix_contract"] is not None
    assert post_state["bnb_horizon"] == 7


def test_bisr_balls_in_bins_dpdl_handoff_stays_minimal_and_opacus_resolves(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre_state, post_state, _sampling_semantics, _dataset_size = _capture_dp_handoff(
        image_dataset_path,
        monkeypatch,
        noise_mechanism="bisr",
        use_explicit_coeffs=False,
        total_steps=7,
    )

    for key in (
        "bnb_accountant_coeffs",
        "bnb_accountant_coeffs_source",
        "bnb_c_matrix",
        "bnb_c_matrix_contract",
    ):
        assert key not in pre_state

    assert pre_state["bnb_horizon"] == 7
    assert post_state["bnb_accountant_coeffs_source"] == "abs_factor_c_col"
    assert post_state["bnb_c_matrix"] is not None
    assert post_state["bnb_c_matrix_contract"] is not None
    assert post_state["bnb_horizon"] == 7


def test_bifr_balls_in_bins_dpdl_handoff_stays_minimal_and_opacus_resolves(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre_state, post_state, _sampling_semantics, _dataset_size = _capture_dp_handoff(
        image_dataset_path,
        monkeypatch,
        noise_mechanism="bifr",
        use_explicit_coeffs=False,
        total_steps=7,
    )

    for key in (
        "bnb_accountant_coeffs",
        "bnb_accountant_coeffs_source",
        "bnb_c_matrix",
        "bnb_c_matrix_contract",
    ):
        assert key not in pre_state

    assert pre_state["bnb_horizon"] == 7
    assert pre_state["bifr_frac"] == pytest.approx(0.25)
    assert post_state["bnb_accountant_coeffs_source"] == "abs_exact_factor_c_col"
    assert post_state["bnb_c_matrix"] is not None
    assert post_state["bnb_c_matrix_contract"] is not None
    assert post_state["bnb_horizon"] == 7


def test_bsr_b_min_sep_dpdl_handoff_defaults_probability_and_opacus_resolves(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre_state, post_state, sampling_semantics, dataset_size = _capture_dp_handoff(
        image_dataset_path,
        monkeypatch,
        noise_mechanism="bsr",
        use_explicit_coeffs=True,
        total_steps=7,
        sampling_mode="b_min_sep",
        bnb_b=4,
    )

    for key in (
        "bnb_accountant_coeffs",
        "bnb_accountant_coeffs_source",
        "bnb_c_matrix",
        "bnb_c_matrix_contract",
    ):
        assert key not in pre_state

    assert pre_state["bnb_horizon"] == 7
    assert sampling_semantics is not None
    assert sampling_semantics.sampling_mode == "b_min_sep"
    p0 = 4.0 / float(dataset_size)
    expected_p = p0 / (1.0 - p0 * 3.0)
    assert abs(float(sampling_semantics.privacy_metadata["p"]) - expected_p) < 1e-12
    assert post_state["bnb_horizon"] == 7


def test_blt_fixed_batch_dpdl_handoff_uses_workload_resolved_state(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    captured: dict[str, dict] = {}
    orig_make_private = opacus.PrivacyEngine.make_private

    def wrapped_make_private(self, *args, **kwargs):
        captured["kwargs"] = copy.deepcopy(kwargs)
        captured["pre_state"] = copy.deepcopy(kwargs["noise_mechanism_config"].mechanism_state)
        return orig_make_private(self, *args, **kwargs)

    monkeypatch.setattr(opacus.PrivacyEngine, "make_private", wrapped_make_private)

    cfg_mgr = ConfigurationManager(
        {
            "command": "train",
            "device": "cpu",
            "dataset_name": "local-image",
            "dataset_path": str(image_dataset_path),
            "model_name": "vit_tiny_patch16_224.augreg_in21k",
            "privacy": True,
            "use_steps": True,
            "total_steps": 7,
            "batch_size": 4,
            "physical_batch_size": 4,
            "num_workers": 0,
            "seed": 42,
            "split_seed": 42,
            "max_grad_norm": 1.0,
            "poisson_sampling": False,
            "target_epsilon": None,
            "noise_batch_ratio": None,
            "noise_mechanism": "blt",
            "accountant": "blt",
            "sampling_mode": "torch_sampler",
            "noise_multiplier": 10.0,
            "blt_buffers": 2,
            "pretrained": False,
        }
    )
    TrainerFactory.get_trainer(cfg_mgr)

    kwargs = captured["kwargs"]
    state = captured["pre_state"]
    assert kwargs["noise_mechanism_config"].mechanism == "blt"
    assert kwargs["noise_mechanism_config"].accounting_mode == "blt_accountant"
    assert kwargs["sampling_semantics"].sampling_mode == "torch_sampler"
    assert state["blt_buffers"] == 2
    assert state["blt_horizon"] == 7
    assert state["blt_selection_mode"] == "implicit_workload_default"
    assert state["noise_multiplier_ref"] == pytest.approx(10.0)
    assert state["z_std"] > 0.0
    assert "forward" in state and "inverse" in state


def test_blt_balls_in_bins_dpdl_handoff_stays_minimal_and_opacus_resolves(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre_state, post_state, sampling_semantics, _dataset_size = _capture_dp_handoff(
        image_dataset_path,
        monkeypatch,
        noise_mechanism="blt",
        use_explicit_coeffs=False,
        total_steps=7,
        sampling_mode="balls_in_bins",
        blt_buffers=2,
    )

    for key in (
        "bnb_accountant_coeffs",
        "bnb_accountant_coeffs_source",
        "bnb_c_matrix",
        "bnb_c_matrix_contract",
    ):
        assert key not in pre_state

    assert sampling_semantics is not None
    assert sampling_semantics.sampling_mode == "balls_in_bins"
    assert pre_state["blt_buffers"] == 2
    assert pre_state["bnb_horizon"] == 7
    assert post_state["bnb_accountant_coeffs_source"] == "normalized_forward_c_col"
    assert post_state["bnb_c_matrix"] is not None
    assert post_state["bnb_c_matrix_contract"] is not None
    assert post_state["bnb_horizon"] == 7


def test_blt_balls_in_bins_target_epsilon_forwards_bnb_controls(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    captured: dict[str, dict] = {}

    def fake_make_private_with_epsilon(self, *args, **kwargs):
        captured["kwargs"] = copy.deepcopy(kwargs)
        return kwargs["module"], kwargs["optimizer"], kwargs["data_loader"]

    monkeypatch.setattr(opacus.PrivacyEngine, "make_private_with_epsilon", fake_make_private_with_epsilon)

    cfg_mgr = ConfigurationManager(
        {
            "command": "train",
            "device": "cpu",
            "dataset_name": "local-image",
            "dataset_path": str(image_dataset_path),
            "model_name": "vit_tiny_patch16_224.augreg_in21k",
            "privacy": True,
            "use_steps": True,
            "total_steps": 7,
            "batch_size": 4,
            "physical_batch_size": 4,
            "num_workers": 0,
            "seed": 42,
            "split_seed": 42,
            "max_grad_norm": 1.0,
            "poisson_sampling": False,
            "target_epsilon": 8.0,
            "noise_multiplier": None,
            "noise_batch_ratio": None,
            "noise_mechanism": "blt",
            "accountant": "bnb",
            "sampling_mode": "balls_in_bins",
            "bnb_b": 2,
            "bnb_num_samples": 123,
            "bnb_calibration_mode": "optimistic",
            "blt_buffers": 2,
            "pretrained": False,
        }
    )
    TrainerFactory.get_trainer(cfg_mgr)

    kwargs = captured["kwargs"]
    assert kwargs["noise_mechanism_config"].mechanism == "blt"
    assert kwargs["noise_mechanism_config"].accounting_mode == "bnb_accountant"
    assert kwargs["sampling_semantics"].sampling_mode == "balls_in_bins"
    assert kwargs["bnb_num_samples"] == 123
    assert kwargs["bnb_calibration_mode"] == "optimistic"
    assert kwargs["blt_buffers"] == 2


def test_bifr_balls_in_bins_target_epsilon_forwards_bnb_controls(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(
        trainer_mod.opacus.distributed,
        "DifferentiallyPrivateDistributedDataParallel",
        lambda model: model,
    )

    captured: dict[str, dict] = {}

    def fake_make_private_with_epsilon(self, *args, **kwargs):
        captured["kwargs"] = copy.deepcopy(kwargs)
        return kwargs["module"], kwargs["optimizer"], kwargs["data_loader"]

    monkeypatch.setattr(opacus.PrivacyEngine, "make_private_with_epsilon", fake_make_private_with_epsilon)

    cfg_mgr = ConfigurationManager(
        {
            "command": "train",
            "device": "cpu",
            "dataset_name": "local-image",
            "dataset_path": str(image_dataset_path),
            "model_name": "vit_tiny_patch16_224.augreg_in21k",
            "privacy": True,
            "use_steps": True,
            "total_steps": 7,
            "batch_size": 4,
            "physical_batch_size": 4,
            "num_workers": 0,
            "seed": 42,
            "split_seed": 42,
            "max_grad_norm": 1.0,
            "poisson_sampling": False,
            "target_epsilon": 8.0,
            "noise_multiplier": None,
            "noise_batch_ratio": None,
            "noise_mechanism": "bifr",
            "accountant": "bnb",
            "sampling_mode": "balls_in_bins",
            "bnb_b": 2,
            "bnb_num_samples": 123,
            "bnb_calibration_mode": "optimistic",
            "bsr_bands": 2,
            "bifr_frac": 0.25,
            "pretrained": False,
        }
    )
    TrainerFactory.get_trainer(cfg_mgr)

    kwargs = captured["kwargs"]
    assert kwargs["noise_mechanism_config"].mechanism == "bifr"
    assert kwargs["noise_mechanism_config"].accounting_mode == "bnb_accountant"
    assert kwargs["sampling_semantics"].sampling_mode == "balls_in_bins"
    assert kwargs["bnb_num_samples"] == 123
    assert kwargs["bnb_calibration_mode"] == "optimistic"
    assert kwargs["bifr_frac"] == pytest.approx(0.25)


@pytest.mark.integration
def test_integration_train_dp_blt_fixed_noise_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-blt-fixed-noise',
        use_target_epsilon=False,
        total_steps=2,
        sampling_mode='torch_sampler',
        noise_mechanism='blt',
        accountant='blt',
        blt_buffers=2,
    )
    assert 'loss' in metrics


@pytest.mark.integration
def test_integration_train_dp_blt_balls_in_bins_fixed_noise_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-blt-balls-in-bins-fixed-noise',
        use_target_epsilon=False,
        total_steps=2,
        sampling_mode='balls_in_bins',
        noise_mechanism='blt',
        accountant='bnb',
        blt_buffers=2,
    )
    assert 'loss' in metrics


@pytest.mark.integration
def test_integration_train_dp_bifr_balls_in_bins_fixed_noise_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-bifr-balls-in-bins-fixed-noise',
        use_target_epsilon=False,
        total_steps=2,
        sampling_mode='balls_in_bins',
        noise_mechanism='bifr',
        accountant='bnb',
    )
    assert 'loss' in metrics


# XXX: This is EXTREMELY slow. First of all the MC estimation of sigma
#      is slow and then it completely blows up with target epsilon, since
#      each iteration of binary search reques the slow MC estimation.
#@pytest.mark.integration
#def test_integration_train_dp_bnb_target_epsilon_path(
#    tmp_path: Path, image_dataset_path: Path
#) -> None:
#    metrics = _run_dp_bnb(
#        tmp_path,
#        image_dataset_path,
#        experiment='train-dp-bnb-target-epsilon',
#        use_target_epsilon=True,
#    )
#    assert 'loss' in metrics
