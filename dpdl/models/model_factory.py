import logging
import re
import timm
import torch
import os

from typing import Any, Dict, Optional

from .model_base import ModelBase
from .wide_resnet import WideResNet
from .koskela_model import KoskelaNet
from .bsr_test_model import BSRTestModel
from .vgg_bnb_reference_model import VGGBnBReferenceModel
from .hugging_face_models import HuggingfaceLanguageModel

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dpdl.configurationmanager import Configuration, Hyperparameters
from dpdl.peft import PeftFactory

log = logging.getLogger(__name__)

def add_noise_to_weights(model, noise_level):
    for name, param in model.named_parameters():
        if 'weight' in name:
            noise = torch.randn(param.size()) * noise_level
            param.data.add_(noise)

def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint by modification time"""
    if checkpoint_dir is None:
        return None

    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        d
        for d in os.listdir(checkpoint_dir)
        if d.startswith('checkpoint_step_') and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]

    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, latest)

class ModelFactory:

    @staticmethod
    def get_model(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        num_classes: int,
        loss_fn: torch.nn,
        metrics: Optional[Dict[str, Any]] = None
    ):

        """
        Create a model instance based on the configuration, with support for PEFT and zeroing head weights.

        Parameters:
        - configuration: Configuration object containing model specs.
        - hyperparams: Optional hyperparameters, not directly used here.
        - num_classes: The number of classes for a classification problem

        Returns:
        - A tuple of (ModelBase instance, Data Transforms).
        """

        """
        TO DO: create LLM_base class, similar to ModelBase, but for LLMs?
        or just use ModelBase directly?
        """

        transforms = {}  # No default transforms
        model_instance = None


        # Flag to skip image model creation if we load HF
        loaded_hf = False

        # Flag to see if we load a local model already fine tuned
        checkpoints_dir_latest = None

        # check if we want to experiment on LLMs
        if configuration.llm:
            checkpoints_dir_latest = get_latest_checkpoint(
                configuration.checkpoints_dir
            )
            model_instance = HuggingfaceLanguageModel(
                configuration.model_name,
                configuration.load_in_4bit,
                num_labels=num_classes,
                peft=configuration.peft,
                checkpoint_dir=checkpoints_dir_latest,
                task=configuration.task,
            )

            transforms = model_instance.get_transforms()
            loaded_hf = True

        if not loaded_hf:
            if configuration.model_name.startswith('wrn-'):
                # Parse depth and width from model_name, e.g., 'wrn-16-4'
                parts = configuration.model_name.split('-')
                depth, width = int(parts[1]), int(parts[2])
                model_instance = WideResNet(depth=depth, width=width, num_classes=num_classes)
                transforms = model_instance.get_transforms()
            elif configuration.model_name == 'koskela-net':
                model_instance = KoskelaNet()
                transforms = model_instance.get_transforms()
            elif configuration.model_name == 'bsr-test-net':
                model_instance = BSRTestModel(num_classes=num_classes)
                transforms = model_instance.get_transforms()
            elif configuration.model_name in {'bnb-vgg-net', 'vgg_bnb_reference'}:
                model_instance = VGGBnBReferenceModel(
                    num_classes=num_classes,  # Other defaults from the JAX implemenation
                )
                transforms = model_instance.get_transforms()
            else:
                model_instance = timm.create_model(
                    configuration.model_name,
                    pretrained=configuration.pretrained,
                    num_classes=num_classes,
                )

                # Resolve data config and create transforms
                model_config = timm.data.resolve_data_config({}, model=model_instance)
                transforms = timm.data.transforms_factory.create_transform(**model_config)

        # resolve num_classes if needed
        if num_classes is None:
            if hasattr(model_instance, 'config') and getattr(model_instance.config, 'vocab_size', None):
                num_classes = int(model_instance.config.vocab_size)
            elif getattr(model_instance, 'num_classes', None):
                num_classes = int(model_instance.num_classes)
            else:
                raise ValueError('Num classes not given and unable to infer it.')

        # Wrap the instantiated model with ModelBase
        model = ModelBase(
            model_instance=model_instance,
            num_classes=num_classes,
            use_feature_cache=configuration.cache_features,
            criterion=loss_fn,
            metrics=metrics
        )

        # Add noise to (pretrained) weights?
        if configuration.weight_perturbation_level > 0:
            add_noise_to_weights(model, configuration.weight_perturbation_level)

        # zero the head weights?
        if configuration.zero_head:
            model.zero_head_weights()

        # should we do Parameter Efficient Fine-Tuning (PEFT)?
        if configuration.peft:
            model = PeftFactory.get_peft_model(model, configuration, checkpoints_dir_latest)

        return model, transforms, num_classes
