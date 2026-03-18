from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.configurationmanager import Configuration, Hyperparameters
from dpdl.optimizers import OptimizerFactory, PaperSGD


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))


def _make_paper_optimizer(*, alpha: float = 0.9, beta: float = 0.5, lr: float = 0.1) -> tuple[_TinyModel, PaperSGD]:
    model = _TinyModel()
    cfg = Configuration(
        command='train',
        optimizer='paper-sgd',
        optimizer_momentum=beta,
        optimizer_weight_decay=alpha,
    )
    hypers = Hyperparameters(
        learning_rate=lr,
        epochs=1,
        noise_multiplier=None,
        max_grad_norm=None,
        target_epsilon=None,
        noise_batch_ratio=None,
    )
    optimizer = OptimizerFactory.get_optimizer(cfg, hypers, model)
    assert isinstance(optimizer, PaperSGD)
    return model, optimizer


def test_paper_sgd_one_step_matches_closed_form() -> None:
    model, optimizer = _make_paper_optimizer(alpha=0.9, beta=0.5, lr=0.1)
    model.weight.grad = torch.tensor([0.2, -0.4], dtype=torch.float64)

    optimizer.step()

    expected_m = torch.tensor([0.2, -0.4], dtype=torch.float64)
    expected_w = 0.9 * torch.tensor([1.0, -2.0], dtype=torch.float64) - 0.1 * expected_m
    assert torch.allclose(model.weight.detach(), expected_w)
    assert torch.allclose(optimizer.state[model.weight]['momentum_buffer'], expected_m)


def test_paper_sgd_two_steps_preserves_momentum_carry() -> None:
    model, optimizer = _make_paper_optimizer(alpha=0.95, beta=0.8, lr=0.2)

    grad1 = torch.tensor([0.3, -0.1], dtype=torch.float64)
    grad2 = torch.tensor([-0.2, 0.4], dtype=torch.float64)

    model.weight.grad = grad1.clone()
    optimizer.step()
    w1 = 0.95 * torch.tensor([1.0, -2.0], dtype=torch.float64) - 0.2 * grad1
    m1 = grad1

    model.weight.grad = grad2.clone()
    optimizer.step()

    m2 = 0.8 * m1 + grad2
    w2 = 0.95 * w1 - 0.2 * m2
    assert torch.allclose(model.weight.detach(), w2)
    assert torch.allclose(optimizer.state[model.weight]['momentum_buffer'], m2)


def test_paper_sgd_zero_gradient_still_applies_shrink_and_momentum_decay() -> None:
    model, optimizer = _make_paper_optimizer(alpha=0.99, beta=0.5, lr=0.1)

    model.weight.grad = torch.tensor([0.4, -0.2], dtype=torch.float64)
    optimizer.step()
    buffer_after_first = optimizer.state[model.weight]['momentum_buffer'].clone()
    weight_after_first = model.weight.detach().clone()

    model.weight.grad = torch.zeros_like(model.weight)
    optimizer.step()

    expected_m = 0.5 * buffer_after_first
    expected_w = 0.99 * weight_after_first - 0.1 * expected_m
    assert torch.allclose(optimizer.state[model.weight]['momentum_buffer'], expected_m)
    assert torch.allclose(model.weight.detach(), expected_w)


def test_standard_sgd_factory_preserves_pytorch_weight_decay_semantics() -> None:
    model = _TinyModel()
    cfg = Configuration(
        command='train',
        optimizer='SGD',
        optimizer_momentum=0.3,
        optimizer_weight_decay=0.2,
    )
    hypers = Hyperparameters(
        learning_rate=0.05,
        epochs=1,
        noise_multiplier=None,
        max_grad_norm=None,
        target_epsilon=None,
        noise_batch_ratio=None,
    )

    optimizer = OptimizerFactory.get_optimizer(cfg, hypers, model)

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]['momentum'] == pytest.approx(0.3)
    assert optimizer.param_groups[0]['weight_decay'] == pytest.approx(0.2)


def test_paper_sgd_exposes_weight_decay_metadata_for_workload_readers() -> None:
    _, optimizer = _make_paper_optimizer(alpha=0.9999, beta=0.95, lr=0.1)

    assert optimizer.param_groups[0]['momentum'] == pytest.approx(0.95)
    assert optimizer.param_groups[0]['weight_decay'] == pytest.approx(0.9999)
