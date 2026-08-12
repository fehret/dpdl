from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('opacus')

from integration_utils import (
    assert_config_and_hyperparams,
    assert_final_noise_multiplier,
    assert_runtime,
    assert_test_metrics,
    base_env,
    run_distributed,
)


def _run_dp(tmp_path: Path, image_dataset_path: Path, experiment: str, epsilon: float) -> tuple[float, float]:
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
        'vit_tiny_patch16_224.augreg_in21k',
        '--no-pretrained',
        '--privacy',
        '--use-steps',
        '--total-steps',
        '2',
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
        '--target-epsilon',
        str(epsilon),
        '--max-grad-norm',
        '1.0',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        experiment,
    ]

    run_distributed(cmd_args, env, repo_root)

    assert_config_and_hyperparams(
        tmp_path / experiment,
        expected_config={
            'command': 'train',
            'device': 'cpu',
            'dataset_name': 'local-image',
            'dataset_path': str(image_dataset_path),
            'model_name': 'vit_tiny_patch16_224.augreg_in21k',
            'privacy': True,
            'use_steps': True,
            'log_dir': str(tmp_path),
            'experiment_name': experiment,
            'seed': 42,
            'split_seed': 42,
        },
        expected_hyperparams={
            'epochs': None,
            'total_steps': 2,
            'batch_size': 4,
            'target_epsilon': float(epsilon),
            'max_grad_norm': 1.0,
        },
    )

    metrics = assert_test_metrics(tmp_path / experiment)
    assert_runtime(tmp_path / experiment)
    noise_multiplier = assert_final_noise_multiplier(tmp_path / experiment)
    return metrics['loss'], noise_multiplier


@pytest.mark.integration
def test_dp_lower_epsilon_higher_loss(tmp_path: Path, image_dataset_path: Path) -> None:
    # Higher epsilon (weaker privacy) should not yield worse loss than lower epsilon.
    loss_high_eps, noise_high_eps = _run_dp(tmp_path, image_dataset_path, 'dp-eps-8', epsilon=8)
    loss_low_eps, noise_low_eps = _run_dp(tmp_path, image_dataset_path, 'dp-eps-2', epsilon=2)

    assert loss_low_eps >= loss_high_eps - 1e-6

    # Same goes for the noise multiplier
    assert noise_low_eps > noise_high_eps
