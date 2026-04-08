from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.configurationmanager import Configuration


def test_bandmf_cyclic_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bandmf',
        accountant='bandmf',
        poisson_sampling=False,
        sampling_mode='cyclic_poisson',
        bsr_bands=8,
    )
    assert cfg.noise_mechanism == 'bandmf'
    assert cfg.sampling_mode == 'cyclic_poisson'


def test_bandmf_fixed_batch_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bandmf',
        accountant='bandmf',
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        bsr_bands=4,
        bsr_coeffs=[1.0, 0.2],
    )
    assert cfg.noise_mechanism == 'bandmf'
    assert cfg.sampling_mode == 'torch_sampler'


def test_bandmf_cyclic_rejects_fixed_batch_knobs() -> None:
    with pytest.raises(ValidationError, match='fixed-batch BSR only'):
        Configuration(
            command='train',
            noise_mechanism='bandmf',
            accountant='bandmf',
            poisson_sampling=False,
            sampling_mode='cyclic_poisson',
            bsr_bands=8,
            bsr_mf_sensitivity=1.0,
        )


def test_bsr_cyclic_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='cyclic_poisson',
        bsr_bands=8,
    )
    assert cfg.noise_mechanism == 'bsr'
    assert cfg.sampling_mode == 'cyclic_poisson'


def test_bisr_cyclic_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bisr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='cyclic_poisson',
        bsr_bands=8,
    )
    assert cfg.noise_mechanism == 'bisr'
    assert cfg.sampling_mode == 'cyclic_poisson'


def test_bandinvmf_cyclic_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bandinvmf',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='cyclic_poisson',
        bsr_bands=8,
    )
    assert cfg.noise_mechanism == 'bandinvmf'
    assert cfg.sampling_mode == 'cyclic_poisson'


def test_fixed_batch_bsr_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        bsr_bands=4,
        bsr_coeffs=[1.0, 0.2],
    )
    assert cfg.noise_mechanism == 'bsr'
    assert cfg.sampling_mode == 'torch_sampler'


def test_fixed_batch_bisr_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bisr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        bsr_bands=4,
        bsr_coeffs=[1.0, 0.2],
    )
    assert cfg.noise_mechanism == 'bisr'
    assert cfg.sampling_mode == 'torch_sampler'


def test_fixed_batch_bandinvmf_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bandinvmf',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        bsr_bands=4,
        bsr_coeffs=[1.0, 0.2],
    )
    assert cfg.noise_mechanism == 'bandinvmf'
    assert cfg.sampling_mode == 'torch_sampler'


def test_fixed_batch_bifr_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bifr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        bsr_bands=4,
        bifr_frac=0.25,
    )
    assert cfg.noise_mechanism == 'bifr'
    assert cfg.sampling_mode == 'torch_sampler'
    assert cfg.bifr_frac == pytest.approx(0.25)


def test_fixed_batch_blt_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='blt',
        accountant='blt',
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        blt_rank=2,
    )
    assert cfg.noise_mechanism == 'blt'
    assert cfg.accountant == 'blt'
    assert cfg.blt_rank == 2


def test_amplified_blt_balls_in_bins_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='blt',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        blt_rank=2,
    )
    assert cfg.noise_mechanism == 'blt'
    assert cfg.accountant == 'bnb'
    assert cfg.sampling_mode == 'balls_in_bins'


def test_blt_rejects_lambda_surface() -> None:
    from dpdl.configurationmanager import ConfigurationManager

    with pytest.raises(ValueError, match='lambda'):
        ConfigurationManager(
            {
                'command': 'train',
                'noise_mechanism': 'blt',
                'accountant': 'blt',
                'poisson_sampling': False,
                'sampling_mode': 'torch_sampler',
                'target_epsilon': 4.0,
                'target_delta': 1e-5,
                'lambda': 0.4,
            }
        )


def test_fixed_batch_bsr_cyclic_is_explicitly_allowed() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='cyclic_poisson',
        bsr_bands=4,
    )
    assert cfg.noise_mechanism == 'bsr'
    assert cfg.sampling_mode == 'cyclic_poisson'


def test_bisr_rejects_wrong_accountant() -> None:
    with pytest.raises(ValidationError, match='BISR mechanism requires --accountant in \\{bnb, bsr\\}'):
        Configuration(
            command='train',
            noise_mechanism='bisr',
            accountant='prv',
            poisson_sampling=False,
            sampling_mode='cyclic_poisson',
            bsr_bands=4,
        )


def test_bisr_rejects_unsupported_sampling_mode() -> None:
    with pytest.raises(ValidationError, match='requires --accountant bnb'):
        Configuration(
            command='train',
            noise_mechanism='bisr',
            accountant='bsr',
            poisson_sampling=False,
            sampling_mode='balls_in_bins',
            bsr_bands=4,
        )


def test_bifr_rejects_wrong_accountant() -> None:
    with pytest.raises(ValidationError, match='BIFR mechanism requires --accountant in \\{bsr\\}'):
        Configuration(
            command='train',
            noise_mechanism='bifr',
            accountant='bnb',
            poisson_sampling=False,
            sampling_mode='torch_sampler',
            bsr_bands=4,
        )


def test_bifr_rejects_unsupported_sampling_mode() -> None:
    with pytest.raises(ValidationError, match='does not support --sampling-mode'):
        Configuration(
            command='train',
            noise_mechanism='bifr',
            accountant='bsr',
            poisson_sampling=False,
            sampling_mode='cyclic_poisson',
            bsr_bands=4,
        )


def test_bandmf_valid_balls_in_bins_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bandmf',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        bnb_b=4,
        bsr_bands=2,
        bsr_coeffs=[1.0, 0.2],
    )
    assert cfg.noise_mechanism == 'bandmf'
    assert cfg.accountant == 'bnb'
    assert cfg.sampling_mode == 'balls_in_bins'


def test_bsr_valid_balls_in_bins_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        bnb_b=4,
        bsr_bands=2,
    )
    assert cfg.noise_mechanism == 'bsr'
    assert cfg.accountant == 'bnb'
    assert cfg.sampling_mode == 'balls_in_bins'


def test_bnb_calibration_controls_are_accepted() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        bnb_b=4,
        bsr_bands=2,
        bnb_num_samples=12345,
        bnb_chunk_size=1000,
        bnb_seed=7,
        bnb_calibration_mode='optimistic',
    )
    assert cfg.bnb_num_samples == 12345
    assert cfg.bnb_chunk_size == 1000
    assert cfg.bnb_seed == 7
    assert cfg.bnb_calibration_mode == 'optimistic'


def test_bnb_calibration_mode_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError, match='bnb_calibration_mode'):
        Configuration(
            command='train',
            noise_mechanism='bsr',
            accountant='bnb',
            poisson_sampling=False,
            sampling_mode='balls_in_bins',
            bnb_b=4,
            bsr_bands=2,
            bnb_calibration_mode='bad-mode',
        )


def test_empty_bsr_coeff_list_normalizes_to_none() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        bnb_b=4,
        bsr_bands=2,
        bsr_coeffs=[],
    )
    assert cfg.bsr_coeffs is None


def test_bisr_valid_balls_in_bins_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bisr',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        bnb_b=4,
        bsr_bands=2,
    )
    assert cfg.noise_mechanism == 'bisr'
    assert cfg.accountant == 'bnb'
    assert cfg.sampling_mode == 'balls_in_bins'


def test_bandinvmf_valid_balls_in_bins_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bandinvmf',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        bnb_b=4,
        bsr_bands=2,
    )
    assert cfg.noise_mechanism == 'bandinvmf'
    assert cfg.accountant == 'bnb'
    assert cfg.sampling_mode == 'balls_in_bins'


def test_balls_in_bins_mf_accepts_missing_bnb_b() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        bsr_bands=4,
    )
    assert cfg.sampling_mode == 'balls_in_bins'
    assert cfg.bnb_b is None


def test_gaussian_bnb_valid_balls_in_bins_with_alias() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='gaussian',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_n_bins',
        bnb_b=4,
    )
    assert cfg.sampling_mode == 'balls_in_bins'


def test_gaussian_bnb_accepts_b_min_sep_without_eager_defaulted_p() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='gaussian',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='b_min_sep',
        bnb_b=4,
    )
    assert cfg.sampling_mode == 'b_min_sep'
    assert cfg.bnb_p is None


def test_bsr_bnb_accepts_b_min_sep_without_explicit_bnb_b() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='b_min_sep',
        bsr_bands=4,
    )
    assert cfg.sampling_mode == 'b_min_sep'
    assert cfg.bnb_b is None


def test_gaussian_bnb_accepts_torch_sampler_configuration() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='gaussian',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='torch_sampler',
    )
    assert cfg.noise_mechanism == 'gaussian'
    assert cfg.accountant == 'bnb'


def test_paper_sgd_requires_optimizer_momentum_and_weight_decay() -> None:
    with pytest.raises(ValidationError, match='paper-sgd requires --optimizer-momentum'):
        Configuration(
            command='train',
            optimizer='paper-sgd',
            optimizer_weight_decay=0.9999,
        )

    with pytest.raises(ValidationError, match='paper-sgd requires --optimizer-weight-decay'):
        Configuration(
            command='train',
            optimizer='paper-sgd',
            optimizer_momentum=0.95,
        )


def test_paper_sgd_rejects_zero_weight_decay() -> None:
    with pytest.raises(ValidationError, match='paper shrinkage factor alpha'):
        Configuration(
            command='train',
            optimizer='paper-sgd',
            optimizer_momentum=0.95,
            optimizer_weight_decay=0.0,
        )


def test_paper_sgd_accepts_matching_configuration() -> None:
    cfg = Configuration(
        command='train',
        optimizer='paper-sgd',
        optimizer_momentum=0.95,
        optimizer_weight_decay=0.9999,
    )
    assert cfg.optimizer == 'paper-sgd'
    assert cfg.optimizer_weight_decay == pytest.approx(0.9999)
    assert cfg.optimizer_momentum == pytest.approx(0.95)


def test_gaussian_rejects_bsr_accountant() -> None:
    with pytest.raises(ValidationError, match='Gaussian mechanism does not support mechanism-specific accountants'):
        Configuration(
            command='train',
            noise_mechanism='gaussian',
            accountant='bsr',
        )


def test_gaussian_accepts_bnb_balls_in_bins_contract() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='gaussian',
        accountant='bnb',
        sampling_mode='balls_in_bins',
        poisson_sampling=False,
        bnb_b=8,
    )
    assert cfg.accountant == 'bnb'
