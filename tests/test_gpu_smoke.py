from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from integration_utils import assert_runtime, assert_test_metrics, base_env, run_distributed


def _should_skip_gpu_tests() -> bool:
    if not torch.cuda.is_available():
        return True
    try:
        x = torch.zeros(1, device='cuda')
        x.add_(1.0)
    except Exception:
        return True
    return False


@pytest.mark.gpu
def test_smoke_train_non_dp(tmp_path: Path, image_dataset_path: Path) -> None:
    if _should_skip_gpu_tests():
        pytest.skip('CUDA not available.')

    repo_root = Path(__file__).resolve().parents[1]
    env = base_env()

    cmd_args = [
        'run.py',
        'train',
        '--device',
        'cuda',
        '--dataset-name',
        'local-image',
        '--dataset-path',
        str(image_dataset_path),
        '--model-name',
        'resnet18',
        '--no-pretrained',
        '--no-privacy',
        '--use-steps',
        '--total-steps',
        '2',
        '--batch-size',
        '4',
        '--physical-batch-size',
        '4',
        '--num-workers',
        '0',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'smoke-non-dp',
    ]

    run_distributed(cmd_args, env, repo_root)
    assert_test_metrics(tmp_path / 'smoke-non-dp')
    assert_runtime(tmp_path / 'smoke-non-dp')


@pytest.mark.gpu
def test_smoke_train_dp(tmp_path: Path, image_dataset_path: Path) -> None:
    if _should_skip_gpu_tests():
        pytest.skip('CUDA not available.')

    repo_root = Path(__file__).resolve().parents[1]
    env = base_env()

    cmd_args = [
        'run.py',
        'train',
        '--device',
        'cuda',
        '--dataset-name',
        'local-image',
        '--dataset-path',
        str(image_dataset_path),
        '--model-name',
        'resnet18',
        '--no-pretrained',
        '--use-steps',
        '--total-steps',
        '2',
        '--batch-size',
        '4',
        '--physical-batch-size',
        '4',
        '--num-workers',
        '0',
        '--noise-multiplier',
        '1.0',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'smoke-dp',
    ]

    run_distributed(cmd_args, env, repo_root)
    assert_test_metrics(tmp_path / 'smoke-dp')
    assert_runtime(tmp_path / 'smoke-dp')
