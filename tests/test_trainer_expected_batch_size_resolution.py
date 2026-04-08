import pytest

pytest.importorskip('torch')
pytest.importorskip('opacus')

from dpdl.trainer import DifferentiallyPrivateTrainer


def test_expected_batch_size_resolution_total_steps_b_min_sep() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=100,
        poisson_sampling=False,
        sampling_mode='b_min_sep',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=0.25,
        bnb_b=2,
    )
    assert got == 12


def test_expected_batch_size_resolution_total_steps_balls_in_bins() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=100,
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=8,
    )
    assert got == 8


def test_expected_batch_size_resolution_total_steps_balls_in_bins_defaults_bins_from_steps_per_epoch() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=100,
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=None,
    )
    assert got == 4


def test_expected_batch_size_resolution_total_steps_torch_sampler_bsr() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=100,
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=None,
    )
    assert got == 4


def test_expected_batch_size_resolution_epochs_path() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=None,
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=None,
    )
    assert got == 4


def test_expected_batch_size_resolution_requires_sampler_inputs() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=10,
        poisson_sampling=False,
        sampling_mode='b_min_sep',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=2,
    )
    assert got == 4

    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=10,
        poisson_sampling=False,
        sampling_mode='b_min_sep',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=None,
        bsr_bands=4,
    )
    assert got == 4

    with pytest.raises(ValueError, match='requires bnb_b or a derivable band parameter'):
        DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
            total_steps=10,
            poisson_sampling=False,
            sampling_mode='b_min_sep',
            batch_size=4,
            dataset_size=64,
            dataloader_len=16,
            bnb_p=None,
            bnb_b=None,
        )

    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=10,
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=None,
    )
    assert got == 4
