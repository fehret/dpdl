from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.configurationmanager import ConfigurationManager


def test_configuration_manager_requires_explicit_privacy_target_path() -> None:
    with pytest.raises(ValueError, match='requires one explicit target path'):
        ConfigurationManager(
            {
                'command': 'train',
                'privacy': True,
                'target_epsilon': None,
                'noise_multiplier': None,
                'noise_batch_ratio': None,
                'epochs': 1,
            }
        )


def test_configuration_manager_accepts_explicit_bsr_z_std_privacy_path() -> None:
    manager = ConfigurationManager(
        {
            'command': 'train',
            'privacy': True,
            'noise_mechanism': 'bandmf',
            'accountant': 'bandmf',
            'poisson_sampling': False,
            'sampling_mode': 'cyclic_poisson',
            'bsr_bands': 8,
            'bsr_z_std': 0.02,
            'target_epsilon': None,
            'noise_multiplier': None,
            'noise_batch_ratio': None,
            'epochs': 1,
            'batch_size': 32,
            'max_grad_norm': 1.0,
        }
    )
    assert manager.configuration.bsr_z_std == 0.02


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        (
            {
                'noise_mechanism': 'gaussian',
                'accountant': 'prv',
            },
            'BSR/BandMF/BISR/BandInvMF/BIFR-specific parameters require --noise-mechanism bandmf, bsr, bisr, bandinvmf, or bifr',
        ),
        (
            {
                'target_epsilon': 8.0,
            },
            'cannot be combined with --target-epsilon',
        ),
        (
            {
                'noise_multiplier': 1.0,
            },
            'cannot be combined with --target-epsilon',
        ),
        (
            {
                'noise_batch_ratio': 0.2,
            },
            'cannot be combined with --target-epsilon',
        ),
    ],
)
def test_configuration_manager_bsr_z_std_invalid_combo_matrix(
    overrides: dict,
    error_match: str,
) -> None:
    params = {
        'command': 'train',
        'privacy': True,
        'noise_mechanism': 'bandmf',
        'accountant': 'bandmf',
        'poisson_sampling': False,
        'sampling_mode': 'cyclic_poisson',
        'bsr_bands': 8,
        'bsr_z_std': 0.02,
        'target_epsilon': None,
        'noise_multiplier': None,
        'noise_batch_ratio': None,
        'epochs': 1,
        'batch_size': 32,
        'max_grad_norm': 1.0,
    }
    params.update(overrides)

    with pytest.raises(ValueError, match=error_match):
        ConfigurationManager(params)


@pytest.mark.parametrize(
    ("target_epsilon", "noise_multiplier", "noise_batch_ratio"),
    [
        (1.0, None, None),
        (-1.0, None, None),
        (None, 0.8, None),
        (None, None, 0.2),
    ],
)
def test_configuration_manager_accepts_any_explicit_privacy_target_path(
    target_epsilon: float | None,
    noise_multiplier: float | None,
    noise_batch_ratio: float | None,
) -> None:
    manager = ConfigurationManager(
        {
            'command': 'train',
            'privacy': True,
            'target_epsilon': target_epsilon,
            'noise_multiplier': noise_multiplier,
            'noise_batch_ratio': noise_batch_ratio,
            'epochs': 1,
            'batch_size': 4,
            'max_grad_norm': 1.0,
        }
    )
    assert manager.configuration.privacy is True


def test_configuration_manager_accepts_gaussian_bnb_optimistic_without_bifr_frac() -> None:
    manager = ConfigurationManager(
        {
            'command': 'train',
            'privacy': True,
            'noise_mechanism': 'gaussian',
            'accountant': 'bnb',
            'sampling_mode': 'balls_in_bins',
            'poisson_sampling': False,
            'noise_multiplier': 1.2,
            'target_epsilon': None,
            'noise_batch_ratio': None,
            'bnb_b': 98,
            'bnb_bands': 1,
            'bnb_num_samples': 32,
            'bnb_chunk_size': 32,
            'bnb_calibration_mode': 'optimistic',
            'epochs': 1,
            'batch_size': 512,
            'max_grad_norm': 1.0,
        }
    )
    assert manager.configuration.bnb_calibration_mode == 'optimistic'


def test_configuration_manager_rejects_bsr_z_std_with_target_epsilon() -> None:
    with pytest.raises(ValueError, match='cannot be combined with --target-epsilon'):
        ConfigurationManager(
            {
                'command': 'train',
                'privacy': True,
                'noise_mechanism': 'bandmf',
                'accountant': 'bandmf',
                'poisson_sampling': False,
                'sampling_mode': 'cyclic_poisson',
                'bsr_bands': 8,
                'bsr_z_std': 0.02,
                'target_epsilon': 8.0,
                'noise_multiplier': None,
                'noise_batch_ratio': None,
                'epochs': 1,
                'batch_size': 32,
                'max_grad_norm': 1.0,
            }
        )


def test_configuration_manager_allows_bsr_z_std_in_clip_only_mode() -> None:
    manager = ConfigurationManager(
        {
            'command': 'train',
            'privacy': True,
            'noise_mechanism': 'bandmf',
            'accountant': 'bandmf',
            'poisson_sampling': False,
            'sampling_mode': 'cyclic_poisson',
            'bsr_bands': 8,
            'bsr_z_std': 0.02,
            'target_epsilon': -1.0,
            'noise_multiplier': None,
            'noise_batch_ratio': None,
            'epochs': 1,
            'batch_size': 32,
            'max_grad_norm': 1.0,
        }
    )
    assert manager.configuration.bsr_z_std == 0.02


def test_configuration_manager_rejects_bsr_z_std_with_noise_multiplier() -> None:
    with pytest.raises(ValueError, match='cannot be combined with --target-epsilon'):
        ConfigurationManager(
            {
                'command': 'train',
                'privacy': True,
                'noise_mechanism': 'bandmf',
                'accountant': 'bandmf',
                'poisson_sampling': False,
                'sampling_mode': 'cyclic_poisson',
                'bsr_bands': 8,
                'bsr_z_std': 0.02,
                'target_epsilon': None,
                'noise_multiplier': 1.0,
                'noise_batch_ratio': None,
                'epochs': 1,
                'batch_size': 32,
                'max_grad_norm': 1.0,
            }
        )


def test_configuration_manager_rejects_bsr_z_std_with_noise_batch_ratio() -> None:
    with pytest.raises(ValueError, match='cannot be combined with --target-epsilon'):
        ConfigurationManager(
            {
                'command': 'train',
                'privacy': True,
                'noise_mechanism': 'bandmf',
                'accountant': 'bandmf',
                'poisson_sampling': False,
                'sampling_mode': 'cyclic_poisson',
                'bsr_bands': 8,
                'bsr_z_std': 0.02,
                'target_epsilon': None,
                'noise_multiplier': None,
                'noise_batch_ratio': 0.2,
                'epochs': 1,
                'batch_size': 32,
                'max_grad_norm': 1.0,
            }
        )


def test_configuration_manager_rejects_legacy_bsr_sensitivity_scale_key() -> None:
    with pytest.raises(ValueError, match='bsr_sensitivity_scale'):
        ConfigurationManager(
            {
                'command': 'train',
                'privacy': True,
                'noise_mechanism': 'bandmf',
                'accountant': 'bandmf',
                'poisson_sampling': False,
                'sampling_mode': 'cyclic_poisson',
                'bsr_bands': 8,
                'target_epsilon': 8.0,
                'epochs': 1,
                'batch_size': 32,
                'max_grad_norm': 1.0,
                'bsr_sensitivity_scale': 1.23,
            }
        )
