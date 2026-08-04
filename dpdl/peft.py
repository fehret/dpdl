import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from peft import LoraConfig, PeftModel, get_peft_model

from .configurationmanager import Configuration, Hyperparameters

log = logging.getLogger(__name__)

def get_nb_trainable_parameters(model: torch.nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def print_trainable_modules(model: torch.nn.Module):
    log.info('Trainable modules:')
    for module_name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters()):
            log.info(module_name)


def _get_head_module(model: torch.nn.Module):
    try:
        return model.get_classifier()
    except (AttributeError, NotImplementedError):
        return None


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Compute a breakdown of model parameter counts.

    Definitions:
      P_backbone : parameters in the backbone (base model without head / adapters)
      P_frozen   : frozen parameters of the effective model (P_model - d_train)
      P_model    : parameters of the effective model (backbone + PEFT adapters + head)
      d_peft     : trainable PEFT parameters (d_train - d_head)
      d_head     : trainable parameters in the classification head
      d_train    : all trainable parameters (d_peft + d_head)
    """
    total = 0
    trainable = 0
    for param in model.parameters():
        num = param.numel()
        total += num
        if param.requires_grad:
            trainable += num

    # Identify the classification head, if the model exposes one.
    head_module = _get_head_module(model)
    if head_module is not None:
        head_total = sum(p.numel() for p in head_module.parameters())
        head_trainable = sum(p.numel() for p in head_module.parameters() if p.requires_grad)
    else:
        head_total = 0
        head_trainable = 0

    # PEFT adapter weights (e.g. LoRA) are added on top of the backbone
    adapter_total = sum(p.numel() for name, p in model.named_parameters() if 'lora_' in name)

    # PEFT's retains a copy of the head, which we exclude from reporting, as no computations are ever done with it
    frozen_head_duplicate = head_total - head_trainable

    d_head = head_trainable
    d_train = trainable
    d_peft = d_train - d_head
    p_model = total - frozen_head_duplicate
    p_frozen = p_model - d_train
    p_backbone = total - head_total - adapter_total

    return {
        'P_backbone': p_backbone,
        'P_frozen': p_frozen,
        'P_model': p_model,
        'd_peft': d_peft,
        'd_head': d_head,
        'd_train': d_train,
    }


class PeftFactory:
    @staticmethod
    def get_peft_model(model: torch.nn.Module, configuration: Configuration, checkpoints_dir: str = None):
        if configuration.peft == 'lora':
            lora_rank = configuration.lora_rank

            if checkpoints_dir is not None:
                return LoRA.get_peft_model(
                    model=model,
                    model_name=configuration.model_name,
                    lora_rank=lora_rank,
                    checkpoint_dir=checkpoints_dir,
                    is_trainable=True,
                )
            else:
                return LoRA.get_peft_model(
                    model=model,
                    model_name=configuration.model_name,
                    lora_rank=lora_rank,
                    checkpoint_dir=checkpoints_dir,
                    is_trainable=True
                )

        if configuration.peft == 'film':
            return FiLM.get_peft_model(model, configuration.model_name)

        if configuration.peft == 'head-only':
            return HeadOnly.get_peft_model(model, configuration.model_name)

        raise RuntimeError(f'Unkown PEFT method: {configuration.peft}')


class HeadOnly:
    @staticmethod
    def get_peft_model(model: torch.nn.Module, model_name: str):
        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # enable gradients for the head
        model.get_classifier().requires_grad_(True)

        trainable_params, all_params = get_nb_trainable_parameters(model)

        if torch.distributed.get_rank() == 0:
            print_trainable_modules(model)

            log.info(f'Finetuning head only - trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}')

        return model


@dataclass
class FilmConfig:
    target_modules: str
    modules_to_save: List[str] = field(default_factory=list)


class FiLM:
    @staticmethod
    def get_peft_model(model: torch.nn.Module, model_name: str):
        # get the FiLM configuration
        film_config = FiLM._get_config(model_name)

        # compile regex pattern for matching target modules
        pattern = re.compile(film_config.target_modules)

        # initially, we'll freeze all params
        for param in model.parameters():
            param.requires_grad = False

        # we enable the gradient for all parameters in matching modules
        for module_name, module in model.named_modules():
            if pattern.match(module_name):
                for name, param in module.named_parameters():
                    param.requires_grad = True

            # also check if this module is explicitly set to be enabled
            for module_to_save in film_config.modules_to_save:
                if module_name.endswith(module_to_save):
                    for name, param in module.named_parameters():
                        param.requires_grad = True

        trainable_params, all_params = get_nb_trainable_parameters(model)

        if torch.distributed.get_rank() == 0:
            print_trainable_modules(model)

            log.info(f'FiLM setup done - trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}')

        return model

    @staticmethod
    def _get_config(model_name: str):
        if model_name.startswith('vit_') or model_name.startswith('deit_'):
            return FilmConfig(
                target_modules=r'.*\.norm\d?',
                modules_to_save=['model.head'],
            )

        if model_name.startswith('resnetv2_50x1_bit'):
            return FilmConfig(
                target_modules=r'.*\.norm\d$',
                modules_to_save=['model.head.fc'],
            )

        if model_name.startswith('poolformer_'):
            return FilmConfig(
                target_modules=r'.*\.norm\d?',
                modules_to_save=['model.head.fc'],
            )

        if model_name.startswith('convnext_'):
            return FilmConfig(
                target_modules=r'.*\.norm|model.*\.stem\.1',
                modules_to_save=['model.head.fc'],
            )

        if 'distilbert' in model_name:
            return FilmConfig(
                target_modules=r'.*LayerNorm$|.*layer_norm$',
                modules_to_save=['pre_classifier', 'classifier'],
            )

        raise RuntimeError(f'No known FiLM configuration for model: {model_name}')


class LoRA:
    @staticmethod
    def get_peft_model(
        model: torch.nn.Module,
        model_name: str,
        lora_rank: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        is_trainable: Optional[bool] = False,
    ):
        if checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                raise FileNotFoundError(f'Checkpoint directory not found: {checkpoint_dir}')

            lora_model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=is_trainable)
        else:
            lora_config = LoRA._get_config(model_name, lora_rank)
            lora_model = get_peft_model(model, lora_config)

        trainable_params, all_params = get_nb_trainable_parameters(lora_model)

        if torch.distributed.get_rank() == 0:
            print_trainable_modules(model)
            log.info(
                f'LoRA setup done - trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}'
            )

        return lora_model

    @staticmethod
    def _get_config(model_name: str, lora_rank: int = None):
        # default rank
        if not lora_rank:
            lora_rank = 4

        # general recommendation for alpha is 2*rank
        lora_alpha = lora_rank

        if model_name.startswith('vit_'):

            # https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
            # config = LoraConfig(
            #     r=16,
            #     lora_alpha=16,
            #     target_modules=["query", "value"],
            #     lora_dropout=0.1,
            #     bias="none",
            #     modules_to_save=["classifier"],
            # )
            return LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                bias='none',
                target_modules=r'.*\.attn\.qkv',
                modules_to_save=['head'],
            )
        elif model_name.startswith('resnetv2_50x1_bit'):
            return LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                bias='none',
                target_modules=r'stem\.conv|.*\.downsample\.conv|.*\.conv\d',
                modules_to_save=['head.fc'],
            )
        elif 'distilbert' in model_name:
            return LoraConfig(
                task_type='SEQ_CLS',
                r=16,
                lora_alpha=32,
                target_modules=['q_lin', 'v_lin'],  # DistilBERT attention projections
                lora_dropout=0.1,
                bias='none',
            )
        elif 'bert' in model_name:  # For the LLM experiments
            return LoraConfig(
                task_type='SEQ_CLS',
                r=16,  # rank
                lora_alpha=32,
                target_modules=['query', 'value'],
                lora_dropout=0.1,
                bias='none',
            )
        elif 'gpt' in model_name:
            # Configure LoRA for causal LM
            return LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=[
                    'c_attn',
                    'c_proj',
                ],  # For GPT-2, the attention layers are called "c_attn"
                # This is more for LLAMA models
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                inference_mode=False,
                bias='none',
                task_type='CAUSAL_LM',
            )

        raise RuntimeError(f'No known LoRA configuration for model: {model_name}')
