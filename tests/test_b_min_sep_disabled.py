from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.configurationmanager import Configuration


def test_b_min_sep_gaussian_is_accepted_with_explicit_p() -> None:
    cfg = Configuration(
        command="train",
        noise_mechanism="gaussian",
        accountant="bnb",
        poisson_sampling=False,
        sampling_mode="b_min_sep",
        bnb_b=2,
        bnb_p=0.2,
    )
    assert cfg.sampling_mode == "b_min_sep"
    assert cfg.bnb_b == 2
    assert abs(float(cfg.bnb_p) - 0.2) < 1e-12


def test_b_min_sep_gaussian_keeps_p_unset_until_runtime() -> None:
    cfg = Configuration(
        command="train",
        noise_mechanism="gaussian",
        accountant="bnb",
        poisson_sampling=False,
        sampling_mode="b_min_sep",
        bnb_b=4,
    )
    assert cfg.sampling_mode == "b_min_sep"
    assert cfg.bnb_b == 4
    assert cfg.bnb_p is None


def test_b_min_sep_rejects_missing_b() -> None:
    with pytest.raises(ValidationError, match="b_min_sep sampling requires --bnb-b"):
        Configuration(
            command="train",
            noise_mechanism="gaussian",
            accountant="bnb",
            poisson_sampling=False,
            sampling_mode="b_min_sep",
        )


def test_b_min_sep_rejects_non_bnb_accountant() -> None:
    with pytest.raises(ValidationError, match="requires --accountant bnb"):
        Configuration(
            command="train",
            noise_mechanism="gaussian",
            accountant="prv",
            poisson_sampling=False,
            sampling_mode="b_min_sep",
            bnb_b=2,
            bnb_p=0.2,
        )
