from dataclasses import dataclass
from typing import Optional, Dict
from torchmetrics.text import Perplexity

import logging
import torch
import torchmetrics

log = logging.getLogger(__name__)

def _get_classification_metrics(
    additional_metrics: list,
    num_classes: int,
    sync: bool,
    with_confusion_matrix: bool,
) -> torchmetrics.MetricCollection:

    defaults = {'MulticlassAccuracy', 'MulticlassAccuracyWithMicro', 'MulticlassAccuracyPerClass'}
    additional_metrics = set(additional_metrics) - defaults
    
    # NB: If `sync_on_compute` is enabled, this breaks
    # distributed training. If this needs to be enabled,
    # then we also need to actually run the validation on
    # all the GPUs.
    metrics = {
        'MulticlassAccuracy': torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes,
            average='macro',
            sync_on_compute=sync,
        ),
        'MulticlassAccuracyWithMicro': torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes,
            average='micro',
            sync_on_compute=sync,
        ),
        'MulticlassAccuracyPerClass': torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes,
            average='none',
            sync_on_compute=sync,
        ),
    }

    
    custom_metrics = dict([(k, getattr(torchmetrics.classification, k)(num_classes=num_classes, sync_on_compute=sync)) for k in additional_metrics])
    
    metrics.update(custom_metrics)    

    if with_confusion_matrix:
        metrics['ConfusionMatrix'] = torchmetrics.ConfusionMatrix(
            task='multiclass' if num_classes > 2 else 'binary',
            num_classes=num_classes,
            sync_on_compute=sync,
        )

    return torchmetrics.MetricCollection(metrics)


class LanguageModelMetrics(torchmetrics.MetricCollection):
    def __init__(self, vocab_size: int, ignore_index: int, sync: bool) -> None:
        metrics = {
            'MulticlassAccuracy': torchmetrics.classification.MulticlassAccuracy(
                num_classes=vocab_size,
                average='micro',
                ignore_index=ignore_index,
                sync_on_compute=sync,
            ),
            'Perplexity': Perplexity(
                ignore_index=ignore_index,
                sync_on_compute=sync,
            ),
        }
        super().__init__(metrics)

    def update(self, preds, target) -> None:
        # Accuracy metrics use standard flattened inputs
        if not hasattr(preds, 'ndim'):
            return super().update(preds, target)

        # We need to shape the data for perplexity that expects 3D logits and 2D labels
        if preds.ndim == 3:
            shift_logits = preds[:, :-1, :].contiguous()                      # (batch, seq_len-1, vocab)
            shift_labels = target[:, 1:].contiguous()                         # (batch, seq_len-1)
            shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))  # (batch*(seq_len-1), vocab)
            shift_labels_flat = shift_labels.view(-1)                         # (batch*(seq_len-1))

            self['Perplexity'].update(shift_logits, shift_labels)

            for name, metric in self.items():
                if name == 'Perplexity':
                    continue

                metric.update(shift_logits_flat, shift_labels_flat)

            return

        return super().update(preds, target)


def _get_language_model_metrics(
    vocab_size: int,
    ignore_index: int,
    sync: bool,
) -> torchmetrics.MetricCollection:

    return LanguageModelMetrics(
        vocab_size=vocab_size,
        ignore_index=ignore_index,
        sync=sync,
    )


class MetricsFactory:

    @staticmethod
    def get_metrics(
        configuration,
        num_classes: Optional[int] = None,
    ) -> Dict[str, torchmetrics.MetricCollection]:
        task = configuration.task

        # we only validate on rank 0, so there's no need to
        # synchronize when calculating the metrics.
        train_sync, eval_sync = True, False

        if task in ('ImageClassification', 'SequenceClassification'):
            if torch.distributed.get_rank() == 0:
                log.info(f'Task is "{configuration.task}", initializing classification metrics.')

            if not num_classes or num_classes < 1:
                raise ValueError('num_classes required for classification tasks')

            train = _get_classification_metrics(
                additional_metrics=configuration.metrics,
                num_classes=num_classes,
                sync=train_sync,
                with_confusion_matrix=False,
            )
            valid = _get_classification_metrics(
                additional_metrics=configuration.metrics,
                num_classes=num_classes,
                sync=eval_sync,
                with_confusion_matrix=False,
            )
            test = _get_classification_metrics(
                additional_metrics=configuration.metrics,
                num_classes=num_classes,
                sync=eval_sync,
                with_confusion_matrix=True,
            )

        elif task in ('CausalLM', 'InstructLM'):
            if torch.distributed.get_rank() == 0:
                log.info(f'Task is "{configuration.task}", initializing language model metrics.')

            vocab_size = int(num_classes)
            ignore_index = -100

            train = _get_language_model_metrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=train_sync,
            )
            valid = _get_language_model_metrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=eval_sync,
            )
            test = _get_language_model_metrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=eval_sync,
            )

        else:
            raise ValueError(f'No metrics defined for task: {task}')

        metrics = {'train_metrics': train, 'valid_metrics': valid, 'test_metrics': test}
        return metrics
