<p align="center">
  <img src="images/dpdl-logo.png" alt="DPDL logo" width="300">
</p>

<h1 align="center" alt="Easy experimentation for Differentially Private Deep Learning">
  <b>Experiment framework for Differentially Private Deep Learning</b>
</h1>

## Installation and first steps

### Prerequisites

- Python >= 3.10
- PyTorch (CPU or GPU build appropriate for your system)

### Install from source

Clone the repository:

```bash
git clone https://github.com/DPBayes/dpdl.git
cd ./dpdl
```

Create and activate a virtual environment, then install DPDL.

Note that you might want to use `--system-site-packages`, if you are installing DPDL on your cluster.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# You might want to install PyTorch for your platform/CUDA/ROCm first.
# See https://pytorch.org/get-started/locally/

pip install -e .
```

Some features (`--use-steps` and `--normalize-clipping`) require our fork of Opacus:

```bash
pip install "git+https://github.com/DPBayes/opacus.git"
```

Otherwise, the official Opacus can be installed by

```bash
pip install opacus
```

### Test your installation

Run the CPU-only test suite (uses the fake dataset; no downloads):

```bash
pip install -e ".[test]"
pytest -m "not gpu"
```

To run GPU smoke tests (requires CUDA and a visible GPU):

```bash
pytest -m gpu
```

### Example usage

The entry point is [run.py](run.py) (also installed as the `dpdl` CLI).

At minimum, specify `--epochs` (or `--use-steps` with `--total-steps`).
See also the [detailed description](docs/epochs-vs-steps.md) of the distinction.

Real-world example (CIFAR-10 + ResNetV2; downloads data and weights):

```bash
dpdl train --epochs 10 --dataset-name uoft-cs/cifar10 --model-name resnetv2_50x1_bit.goog_in21k --device auto
```

## How to use?

### Settings

Run `dpdl --help` (or `python run.py --help`).

![](images/usage.png)

### Training

You can use the framework to train models with or without privacy enabled.

The results will be stored in a dedicated experiment folder that is placed in the logs folder.
Read more about this in the [experiment directory documentation](docs/experiment-directory.md).

Training is done in a private manner by default.
The privacy-related command line options are listed in the 'Opacus options' section of the help screen.
To train a model without it, simply add the `--no-privacy` command line switch.

Experiments on varying parameters can be set up by creating a bash script, like so:

```
#!/bin/bash
###############################
### Experiment: model-variation
###############################

# Base configurations
EXPERIMENT="model-variation"
LOG_DIR="./experiments/$EXPERIMENT/data"

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k" "resnetv2_50x1_bit.goog_in21k")

# Loop over configurations
for model in "${MODELS[@]}"
do
    EXPERIMENT_SETTING="${model}"

    python run.py train \
        --model-name $model \
        --dataset-name uoft-cs/cifar10 \
        --subset-size 1.0 \
        --num-classes 10 \
        --batch-size 1024 \
        --target-epsilon 1 \
        --experiment-name $EXPERIMENT_SETTING \
        --log-dir $LOG_DIR \
        --privacy \
        --overwrite-experiment
done
```

### Hyperparameter optimization

DPDL also offers the possibility of optimizing hyperparameters for a given configuration.
See the detailed guide: [docs/hyperparameter-optimization.md](docs/hyperparameter-optimization.md).
Again, the specifics of the [experiment directory](docs/experiment-directory.md) apply.

Simple example (optimize learning rate and batch size):

```
dpdl optimize --target-hypers learning_rate --target-hypers batch_size --n-trials 20 --optuna-config conf/optuna_hypers.conf
```

You can combine this with the aforementioned bash script of course, to run parameter impact studies for optimized hyperparameters.
The bash script displayed above could then be changed to:

```
#!/bin/bash
#########################################
### Experiment: optimized-model-variation
#########################################

# Base configurations
EXPERIMENT="optimized-model-variation"
LOG_DIR="./experiments/$EXPERIMENT/data"
OPTUNA_JOURNAL="$LOG_DIR/optuna.journal"
OPTUNA_CONFIG="conf/optuna_hypers-subset1.0.conf"

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k" "resnetv2_50x1_bit.goog_in21k")

# Loop over configurations
for model in "${MODELS[@]}"
do
    EXPERIMENT_SETTING="${model}"

    

    python run.py optimize \
                    --model-name $model \
                    --dataset-name uoft-cs/cifar10 \
                    --subset-size 1.0 \
                    --num-classes 10 \
                    --batch-size 1024 \
                    --target-hypers epochs \
                    --target-hypers learning_rate \
                    --target-hypers max_grad_norm \
                    --target-epsilon 1 \
                    --n-trials 20 \
                    --optuna-config $OPTUNA_CONFIG \
                    --optuna-target-metric MulticlassAccuracy \
                    --optuna-direction maximize \
                    --experiment-name $EXPERIMENT_NAME \
                    --log-dir $LOG_DIR \
                    --privacy
                    --optuna-journal $OPTUNA_JOURNAL \
                    --overwrite-experiment
done
```

### Creating a Slurm script

We also provide a tool for creating Slurm run scripts for LUMI

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

## High-level architecture & Customization

![DPDL Architecture](images/dpdl-architecture.svg)

### Entry point

The entrypoint [run.py](run.py) provides a CLI using Python's Typer module.

### Command-line interface

The CLI implementation is in [dpdl/cli.py](dpdl/cli.py).

If you need to implement additional command line options, start there.

### Training

The CLI calls the `fit` method of [trainer](dpdl/trainer.py).

A detailed explanation can be found in the [Trainer documentation](docs/trainer.md).

The results are stored in the [experiment directory](docs/experiment-directory.md).

### Hyperparameter optimization

The CLI calls the `optimize_hypers` method of [hyperparameteroptimizer](dpdl/hyperparameteroptimizer.py).

See the detailed guide: [docs/hyperparameter-optimization.md](docs/hyperparameter-optimization.md).

### Callbacks

The system provides a flexible [callback system](docs/callbacks.md).

### Add a new dataset?

Create a new [datamodule](dpdl/datamodules.py).

NB: The code currently should support all Huggingface image datasets by using, for example a `--dataset-name cifar100` command line parameter.

### Add a new model?

Follow the steps detailed in the [model documentation](docs/models.md).

### Add a new optimizer?

Add a new optimizer in [optimizers](dpdl/optimizers.py).

## Something broken or missing?

Check for known issues and fixes in the [troubleshoot docs](docs/troubleshoot.md).

If the issue persists, check if it is a known issue via the issue tracker and create a bug report otherwise.

We also welcome contributions, so if you have a fix for a problem or a useful feature, fork the repo and open a pull request with your changes.

## Acknowledgements

We borrow the callback idea from [fastai](https://github.com/fastai/fastai) and the datamodule idea from [PyTorch Lightning](https://github.com/Lightning-AI/lightning).
