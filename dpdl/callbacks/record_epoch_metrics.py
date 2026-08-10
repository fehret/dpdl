import csv
import json
import logging
import os

from ..utils import tensor_to_python_type
from .base_callback import Callback

log = logging.getLogger(__name__)


class RecordMetricsByEpochCallback(Callback):
    """
    Record the per-epoch metrics for the train and validation splits.

    Scalar metrics go to `<log_dir>/epoch_metrics.csv` (one row per epoch, with
    `train_<metric>` / `val_<metric>` columns).

    Matrix-valued metrics (the confusion matrix) is written to `<log_dir>/epoch_confusion_matrices.json`, 
    keyed by epoch and split. Per-class (1-D) metrics are not recorded.

    Validation entries are only present for epochs on which validation ran
    (see `--validation-frequency`).
    """

    _CSV_FILENAME = 'epoch_metrics.csv'
    _MATRICES_FILENAME = 'epoch_confusion_matrices.json'

    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir

        # epoch -> {metric: scalar}
        self._train_scalars = {}
        self._val_scalars = {}

        # epoch -> {metric: 2-D list}
        self._train_matrices = {}
        self._val_matrices = {}

    def on_train_start(self, trainer):
        if self._is_global_zero():
            os.makedirs(self.log_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, epoch, metrics):
        scalars, matrices = self._split_metrics(metrics)
        self._train_scalars[epoch] = scalars
        if matrices:
            self._train_matrices[epoch] = matrices

    def on_validation_epoch_end(self, trainer, epoch, metrics, loss=None):
        scalars, matrices = self._split_metrics(metrics)
        self._val_scalars[epoch] = scalars
        if matrices:
            self._val_matrices[epoch] = matrices

    def on_train_end(self, trainer):
        if not self._is_global_zero():
            return

        self._write_scalar_csv()
        self._write_matrix_json()

    def _write_scalar_csv(self):
        epochs = sorted(set(self._train_scalars) | set(self._val_scalars))

        rows = []
        for epoch in epochs:
            # 1-indexed to match epoch_losses.csv / epoch_accuracy.csv
            row = {'epoch': epoch + 1}
            for name, value in self._train_scalars.get(epoch, {}).items():
                row[f'train_{name}'] = value
            for name, value in self._val_scalars.get(epoch, {}).items():
                row[f'val_{name}'] = value
            rows.append(row)

        train_cols = sorted({k for r in rows for k in r if k.startswith('train_')})
        val_cols = sorted({k for r in rows for k in r if k.startswith('val_')})
        fieldnames = ['epoch'] + train_cols + val_cols

        path = os.path.join(self.log_dir, self._CSV_FILENAME)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(rows)

        log.info(f'Per-epoch metrics written to {path}')

    def _write_matrix_json(self):
        if not self._train_matrices and not self._val_matrices:
            return

        epochs = sorted(set(self._train_matrices) | set(self._val_matrices))

        data = {}
        for epoch in epochs:
            entry = {}
            if epoch in self._train_matrices:
                entry['train'] = self._train_matrices[epoch]
            if epoch in self._val_matrices:
                entry['validation'] = self._val_matrices[epoch]
            data[epoch + 1] = entry

        path = os.path.join(self.log_dir, self._MATRICES_FILENAME)
        with open(path, 'w') as f:
            json.dump(data, f)

        log.info(f'Per-epoch confusion matrices written to {path}')

    @staticmethod
    def _split_metrics(metrics):
        """Split a metrics dict into scalar metrics and matrix-valued (2-D) ones."""
        metrics = tensor_to_python_type(metrics)

        scalars = {}
        matrices = {}
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                scalars[name] = value
            elif isinstance(value, list) and value and all(isinstance(row, list) for row in value):
                matrices[name] = value
            # else: 1-D list (e.g. a per-class metric) -> not recorded
        return scalars, matrices
