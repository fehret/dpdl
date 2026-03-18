from pathlib import Path
import os
import sys
import copy

import pytest
import torch
import opacus

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
    noise_mechanism: str,
    use_explicit_coeffs: bool,
    total_steps: int,
) -> tuple[dict, dict]:
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
    orig_make_private = opacus.PrivacyEngine.make_private

    def wrapped_make_private(self, *args, **kwargs):
        captured["pre_state"] = copy.deepcopy(kwargs["noise_mechanism_config"].mechanism_state)
        result = orig_make_private(self, *args, **kwargs)
        captured["post_state"] = copy.deepcopy(self.noise_mechanism_config.mechanism_state)
        return result

    monkeypatch.setattr(opacus.PrivacyEngine, "make_private", wrapped_make_private)

    cli_params = {
        "command": "train",
        "device": "cpu",
        "dataset_name": "local-image",
        "dataset_path": str(image_dataset_path),
        "model_name": "vit_tiny_patch16_224.augreg_in21k",
        "privacy": True,
        "use_steps": True,
        "total_steps": total_steps,
        "batch_size": 4,
        "physical_batch_size": 4,
        "num_workers": 0,
        "seed": 42,
        "split_seed": 42,
        "max_grad_norm": 1.0,
        "poisson_sampling": False,
        "target_epsilon": None,
        "noise_batch_ratio": None,
        "noise_mechanism": noise_mechanism,
        "accountant": "bnb",
        "sampling_mode": "balls_in_bins",
        "bnb_b": 2,
        "noise_multiplier": 10.0,
        "bsr_bands": 2,
        "pretrained": False,
    }
    if use_explicit_coeffs:
        cli_params["bsr_coeffs"] = [1.0, 0.2] if noise_mechanism == "bsr" else [1.0, -0.5]

    cfg_mgr = ConfigurationManager(cli_params)
    TrainerFactory.get_trainer(cfg_mgr)

    return captured["pre_state"], captured["post_state"]


def _run_dp_bnb(
    tmp_path: Path,
    image_dataset_path: Path,
    *,
    experiment: str,
    use_target_epsilon: bool,
    sampling_mode: str = 'balls_in_bins',
    noise_mechanism: str = 'gaussian',
    accountant: str = 'bnb',
    model_name: str = 'vit_tiny_patch16_224.augreg_in21k',
    model_args: list[str] | None = None,
    use_explicit_coeffs: bool = True,
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
        '1',
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
        '--bnb-b',
        '2',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        experiment,
    ]

    if noise_mechanism == 'bsr':
        cmd_args.extend(['--bsr-bands', '2'])
        if use_explicit_coeffs:
            cmd_args.extend(['--bsr-coeffs', '1.0', '--bsr-coeffs', '0.2'])
    elif noise_mechanism == 'bisr':
        cmd_args.extend(['--bsr-bands', '2'])
        if use_explicit_coeffs:
            cmd_args.extend(['--bsr-coeffs', '1.0', '--bsr-coeffs', '-0.5'])
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
        'total_steps': 1,
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
    if noise_mechanism == 'gaussian':
        expected_config['bnb_b'] = 2
    else:
        expected_config['bnb_b'] = 2
        expected_config['bsr_bands'] = 2

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
    pre_state, post_state = _capture_dp_handoff(
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
    pre_state, post_state = _capture_dp_handoff(
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
