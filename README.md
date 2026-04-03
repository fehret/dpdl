# Easy experimentation for Differentially Private (DP) Deep Learning

The system requires CUDA. We provide scripts for running in a Slurm environment.

Many of the ideas that we are using come from [fastai](https://github.com/fastai/fastai) and [PyTorch Lightning](https://github.com/Lightning-AI/lightning).

## Experiments

![Experiment documentation is here.](experiments/)

## Installation and usage

### Install dependencies

```bash
pip install torch timm datasets "typer[all]" optuna optuna-integration botorch torchmetrics pydantic peft
pip install -e "git+https://github.com/DPBayes/opacus.git@adaptive_optimizer#egg=opacus"
```

### Command line usage

Entry point is [run.py](blob/vanilla-pytorch-refactor/run.py).

### How to use?

#### Command line help

Start an interactive session.

Load your Python environment, or use pre-installed environment on LUMI (see below).

Run `python run.py`, `python run.py --help`, or `python run.py -h`.

### Creating a Slurm script

There is a tool for creating Slurm run scripts for LUMI

```
$ bin/create-run-script.sh
Usage: bin/create-run-script.sh script_name [options...]

script_name               Name of the script to be created.

Options:
  --help                  Show this help message.
  project                 Slurm project (default: project_462000213).
  partition               Slurm partition (default: standard-g).
  gpus                    Number of GPUs (default: 8).
  time                    Time allocation (default: 1:00:00, 00:15:00 for dev-g).
  mem_per_gpu             Memory per GPU (default: 60G).
  cpus_per_task           Number of CPUs per task (default: 7).

Example:
  bin/create-run-script.sh run.sh project_462000213 small-g 1
```

### Training under DP

Check out [an example](experiments/00-experiment-batch-size-variation/scripts/run.sh)

### Training without DP

Check [an example](experiments/06-few-shot-from-scratch-non-dp/scripts/run.sh)

## Architecture

![DPDL Architecture](images/dpdl-architecture.svg)

### Entry point

The entrypoint [run.py](run.py) provides a CLI using Python's Typer module.

### Command-line interface

The CLI implementation is in [dpdl/cli.py](dpdl/cli.py)

### Training

The CLI calls the `fit` method of [trainer](dpdl/trainer.py) 

### Hyperparameter optimization

The CLI calls the `optimize_hypers` method of [hyperparameteroptimizer](dpdl/hyperparameteroptimizer.py).

The ranges/options for the different hyperparameters is in `conf/optuna_hypers.conf`.

### Callbacks

The system provides a flexible [callback system](dpdl/callbacks.py).

## How to?

### Load pre-installed environment on LUMI

First load CSC PyTorch

```
module use /appl/local/csc/modulefiles/

module load pytorch
```

Then activate the pre-installed environment

```
source /scratch/$PROJECT/venvs/dpdl/bin/activate
```

### Add a new dataset?

Create a new [datamodule](dpdl/datamodules.py).

NB: The code currently should support all Huggingface image datasets by using, for example a `--dataset-name cifar100` command line parameter.

### Add a new model?

Create a new model in `dpdl/models` and add it to [models.py](dpdl/models.py).

### Add a new optimizer?

Add a new optimizer in [optimizers](dpdl/optimizers.py).

## TODO

- [x] Instead of optimizing batch sizes, let's optimize sample rates.
- [x] Repeat experiments with different seeds (do in CLI instead?)
- [x] Log runtime, gpu type, and gpu count
- [x] Properly arrange GPU bindings
- [x] Verify that zeroing the head works
- [x] Use/test torch.compile() (does not work with Opacus)
- [x] Use steps instead of epochs to avoid problems with Opacus sample rate
- [x] Possibility to finetune only the head of a model?
- [x] Fix target delta calculation
- [x] Use image transformations included with the timm models
- [x] Use FiLM adaptor instead of training head/all
- [x] More optimizers? Optimizer as a CLI switch?
- [x] When overwriting an experiment, also remove it from the main Optuna journal
- [x] BO search for learning rate in log space
- [x] Possibility to zero the head weights
- [x] LoRA (Low Rank Adapation) for ViT and ResNet50
- [x] Final training round with train + validation datasets
- [x] Save the optuna study in experiment directory (if we want to try more trials)
- [x] Use test set for final Optuna trial accuracy
- [x] Save experiments to log directory
- [x] Save Optuna study to experiment directory after all trials
- [x] Use DistributedSampler in dataloaders for the non-DP case
- [x] Refactor CIFAR10DataModule as HuggingfaceDataModule or similar
- [x] Possibility to only use a subset of dataset

## Maybe TODO

- [ ] Validation/training loss logging?
- [ ] Learning rate schedulers?
