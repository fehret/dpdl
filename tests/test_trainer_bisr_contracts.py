from pathlib import Path
import sys
import json
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip('torch')
pytest.importorskip('opacus')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opacus.mechanism_contracts import SamplingSemantics
from dpdl.trainer import DifferentiallyPrivateTrainer


def test_log_bsr_trace_emits_for_bisr(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger='dpdl.trainer')

    DifferentiallyPrivateTrainer._log_bsr_trace(
        stage='test',
        sampling_semantics=SamplingSemantics(
            sampling_mode='cyclic_poisson',
            privacy_metadata={'bands': 8},
        ),
        noise_mechanism_config=SimpleNamespace(
            mechanism='bisr',
            accounting_mode='bsr_accountant',
            mechanism_state={'coeffs': [1.0, 0.2], 'bsr_bands': 8},
        ),
        has_target_privacy_params=True,
        noise_multiplier_ref=0.8,
        correlated_denominator=32.0,
        mechanism_kwargs={},
    )

    messages = [r.message for r in caplog.records if r.message.startswith('BSR_TRACE ')]
    assert messages, 'expected BSR_TRACE payload for bisr'
    payload = json.loads(messages[-1].split('BSR_TRACE ', 1)[1])
    assert payload['mechanism'] == 'bisr'
    assert payload['sampling_metadata']['bands'] == 8
    assert payload['mechanism_state']['bsr_bands'] == 8


def test_validate_cyclic_steps_vs_bands_rejects_bisr_when_steps_too_small() -> None:
    with pytest.raises(ValueError, match='require steps >= bands'):
        DifferentiallyPrivateTrainer._validate_cyclic_steps_vs_bands(
            mechanism='bisr',
            sampling_mode='cyclic_poisson',
            total_steps=4,
            epochs=None,
            dataloader_len=10,
            bsr_bands=8,
        )


def test_validate_cyclic_steps_vs_bands_accepts_bisr_when_steps_sufficient() -> None:
    DifferentiallyPrivateTrainer._validate_cyclic_steps_vs_bands(
        mechanism='bisr',
        sampling_mode='cyclic_poisson',
        total_steps=16,
        epochs=None,
        dataloader_len=10,
        bsr_bands=8,
    )


def test_log_bsr_trace_emits_for_bandinvmf(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger='dpdl.trainer')

    DifferentiallyPrivateTrainer._log_bsr_trace(
        stage='test',
        sampling_semantics=SamplingSemantics(
            sampling_mode='cyclic_poisson',
            privacy_metadata={'bands': 8},
        ),
        noise_mechanism_config=SimpleNamespace(
            mechanism='bandinvmf',
            accounting_mode='bsr_accountant',
            mechanism_state={'coeffs': [1.0, 0.2], 'bandinvmf_inv_coeffs': [1.0, -0.2], 'bsr_bands': 8},
        ),
        has_target_privacy_params=True,
        noise_multiplier_ref=0.8,
        correlated_denominator=32.0,
        mechanism_kwargs={},
    )

    messages = [r.message for r in caplog.records if r.message.startswith('BSR_TRACE ')]
    assert messages
    payload = json.loads(messages[-1].split('BSR_TRACE ', 1)[1])
    assert payload['mechanism'] == 'bandinvmf'


def test_log_bsr_trace_emits_for_bifr(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    DifferentiallyPrivateTrainer._log_bsr_trace(
        stage='unit-test',
        sampling_semantics=SimpleNamespace(
            sampling_mode='torch_sampler',
            privacy_metadata={},
        ),
        noise_mechanism_config=SimpleNamespace(
            mechanism='bifr',
            accounting_mode='bsr_accountant',
            mechanism_state={'coeffs': [1.0, 0.2], 'bsr_bands': 8, 'bifr_frac': 0.25},
        ),
        has_target_privacy_params=False,
        noise_multiplier_ref=1.0,
        correlated_denominator=16.0,
        mechanism_kwargs={'bifr_frac': 0.25},
    )
    messages = [record.message for record in caplog.records if record.message.startswith('BSR_TRACE ')]
    assert messages, 'expected BSR_TRACE payload for bifr'
    payload = json.loads(messages[-1].split('BSR_TRACE ', 1)[1])
    assert payload['mechanism'] == 'bifr'
    assert payload['mechanism_state']['bifr_frac'] == pytest.approx(0.25)


def test_validate_cyclic_steps_vs_bands_rejects_bandinvmf_when_steps_too_small() -> None:
    with pytest.raises(ValueError, match='require steps >= bands'):
        DifferentiallyPrivateTrainer._validate_cyclic_steps_vs_bands(
            mechanism='bandinvmf',
            sampling_mode='cyclic_poisson',
            total_steps=4,
            epochs=None,
            dataloader_len=10,
            bsr_bands=8,
        )
