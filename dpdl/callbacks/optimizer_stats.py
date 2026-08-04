import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .base_callback import Callback

log = logging.getLogger(__name__)


class RecordOptimizerStatsCallback(Callback):
    """
    Record gradient- and Adam-related statistics at every optimizer step.

    Notation:
      C : the clipping bound (max_grad_norm).
      g : per-example gradient norm ||g_i||, across all trainable
          parameters

    Columns:
      step                     - global optimizer step index
      gradient_norm_q50/90/95/99 - quantiles of g over the logical batch
      q99_over_c               - gradient_norm_q99 / C
      clipped_fraction         - fraction of examples with g > C
      retained_weight_mean     - mean of min(1, C/g)
      retained_weight_sq_mean  - mean of min(1, C/g)^2
      adam_lr_free_update_rms  - RMS(m_hat / (sqrt(v_hat) + adam_epsilon))
      adam_inv_denom_rms       - RMS(1 / (sqrt(v_hat) + adam_epsilon))

    The two Adam fields are aggregated over all trainable parameters, where
    ``m_hat``/``v_hat`` are the first/second moments from the optimizer state after the step. 
    Gradient statistics require per-example gradients (DP training); Adam statistics require an Adam-family optimizer.
    Fields that cannot be computed are recorded as NaN.

    Distributed training: gradient statistics are computed over a logical batch.
    Each rank owns a shard of the per-example gradients, so we all-gather the per-example norms 
    across ranks before computing the quantiles and means
    """

    _FIELDNAMES = [
        'step',
        'gradient_norm_q50',
        'gradient_norm_q90',
        'gradient_norm_q95',
        'gradient_norm_q99',
        'q99_over_c',
        'clipped_fraction',
        'retained_weight_mean',
        'retained_weight_sq_mean',
        'adam_lr_free_update_rms',
        'adam_inv_denom_rms',
    ]

    # Quantiles reported for the per-example gradient norm g.
    _QUANTILES = (0.50, 0.90, 0.95, 0.99)

    def __init__(self, log_dir: str | Path, max_grad_norm: float):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.C = float(max_grad_norm)

        self._rows = []
        # Per-example gradient norms (for this rank).
        self._current_norms = []

        # Placeholders, set in on_train_start
        self._world_size = 1
        self._device = torch.device('cpu')
        self._collect_gradient_stats = False

    def on_train_start(self, trainer, *args, **kwargs):
        super().on_train_start(trainer, *args, **kwargs)

        self._world_size = torch.distributed.get_world_size()
        self._device = trainer.device

        # check if we are in DP training
        self._collect_gradient_stats = hasattr(trainer.optimizer, 'noise_multiplier')

    def on_train_batch_start(self, *args, **kwargs):
        self._current_norms.clear()

    def on_train_physical_batch_end(self, trainer, *args, **kwargs):
        # Collect per-example gradient norms for samples in this physical batch
        with torch.no_grad():
            sum_squares = None

            for group in trainer.optimizer.param_groups:
                for p in group['params']:
                    grad_sample = getattr(p, 'grad_sample', None)
                    if grad_sample is None or grad_sample.numel() == 0:
                        continue

                    # accumulate only the sum of squares and not the gradients (save memory)
                    flat = grad_sample.view(grad_sample.size(0), -1)
                    sq = flat.pow(2).sum(dim=1)
                    sum_squares = sq if sum_squares is None else sum_squares + sq

            if sum_squares is not None:
                self._current_norms.append(sum_squares.sqrt())

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, *args, **kwargs):
        super().on_train_batch_end(trainer, batch_idx, batch, loss, *args, **kwargs)

        row = {name: float('nan') for name in self._FIELDNAMES}
        row['step'] = self.global_step

        if self._collect_gradient_stats:
            self._record_gradient_stats(row)
        self._record_adam_stats(row, trainer.optimizer)

        self._rows.append(row)
        self._current_norms.clear()

    def _record_gradient_stats(self, row: dict) -> None:
        with torch.no_grad():
            if self._current_norms:
                g_local = torch.cat(self._current_norms).float()
            else:
                g_local = torch.empty(0, dtype=torch.float32, device=self._device)

            # Gather the full logical batch across ranks
            g = self._all_gather_norms(g_local)
            if g.numel() == 0:
                return

            q = torch.tensor(self._QUANTILES, device=g.device, dtype=g.dtype)
            q50, q90, q95, q99 = torch.quantile(g, q).tolist()

            # Per-example retained weight: w = min(1, C/g).
            retained = torch.clamp(self.C / g, max=1.0)

            row['gradient_norm_q50'] = q50
            row['gradient_norm_q90'] = q90
            row['gradient_norm_q95'] = q95
            row['gradient_norm_q99'] = q99
            row['q99_over_c'] = q99 / self.C if self.C > 0 else float('nan')
            row['clipped_fraction'] = (g > self.C).float().mean().item()
            row['retained_weight_mean'] = retained.mean().item()
            row['retained_weight_sq_mean'] = retained.pow(2).mean().item()

    def _all_gather_norms(self, g_local: torch.Tensor) -> torch.Tensor:
        """
        All-gather per-sample norm vectors across ranks and
        return their concatenation. Every rank receives the full vector.

        Must be called by all or no ranks.
        """
        if self._world_size == 1:
            return g_local

        device = g_local.device

        # Exchange each sample count (batch sizes may vary)
        local_size = torch.tensor([g_local.numel()], device=device, dtype=torch.long)
        size_list = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(self._world_size)]
        torch.distributed.all_gather(size_list, local_size)
        sizes = [int(s.item()) for s in size_list]

        max_size = max(sizes)
        if max_size == 0:
            return g_local.new_empty(0)

        # Pad to a common length so all_gather sees equal-sized tensors.
        padded = g_local.new_zeros(max_size)
        if g_local.numel() > 0:
            padded[: g_local.numel()] = g_local

        gathered = [g_local.new_zeros(max_size) for _ in range(self._world_size)]
        torch.distributed.all_gather(gathered, padded)

        # Recover true sizes and combine
        parts = [gathered[i][: sizes[i]] for i in range(self._world_size) if sizes[i] > 0]
        return torch.cat(parts)

    def _record_adam_stats(self, row: dict, optimizer) -> None:
        stats = self._compute_adam_stats(optimizer)
        if stats is None:
            return

        row['adam_lr_free_update_rms'], row['adam_inv_denom_rms'] = stats

    def _compute_adam_stats(self, optimizer) -> Optional[Tuple[float, float]]:
        sum_sq_update = 0.0
        sum_sq_inv_denom = 0.0
        count = 0

        #optimizer state is identical across ranks so no gather is needed
        with torch.no_grad():
            for group in optimizer.param_groups:
                beta1, beta2 = group['betas']
                eps = group['eps']

                for p in group['params']:
                    state = optimizer.state.get(p, {})
                    exp_avg = state.get('exp_avg')
                    exp_avg_sq = state.get('exp_avg_sq')
                    if exp_avg is None or exp_avg_sq is None:
                        continue

                    step_t = state.get('step', 0)
                    if isinstance(step_t, torch.Tensor):
                        step_t = step_t.item()
                    if step_t <= 0:
                        continue

                    bias_correction1 = 1.0 - beta1 ** step_t
                    bias_correction2 = 1.0 - beta2 ** step_t

                    m_hat = exp_avg / bias_correction1
                    second_moment = state.get('max_exp_avg_sq', exp_avg_sq)
                    v_hat = second_moment / bias_correction2

                    denom = v_hat.sqrt() + eps
                    update = m_hat / denom
                    inv_denom = denom.reciprocal()

                    sum_sq_update += update.pow(2).sum().item()
                    sum_sq_inv_denom += inv_denom.pow(2).sum().item()
                    count += update.numel()

        if count == 0:
            return None

        lr_free_update_rms = (sum_sq_update / count) ** 0.5
        inv_denom_rms = (sum_sq_inv_denom / count) ** 0.5
        return lr_free_update_rms, inv_denom_rms

    def on_train_end(self, trainer, *args, **kwargs):
        if not self._rows or not self._is_global_zero():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.log_dir / 'optimizer_stats.csv'

        with out_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._rows)

        log.info('Optimizer stats written to %s', out_path)
