# Flower Federated Learning Experiments

This folder contains the code used to run the federated learning experiments with [Flower (flwr)](https://flower.ai/) described in the paper, using **Multilayer Perceptron (MLP)** and **Logistic Regression (LR)** as learning models.

## Folder Structure

| File | Description |
|------|-------------|
| `create_run_flwr_LR.py` | Generates `.sh` scripts for running federated LR experiments |
| `create_run_flwr_MLP.py` | Generates `.sh` scripts for running federated MLP experiments |
| `load_model.py` | Loads a trained model and evaluates it on the test dataset |
| `*.sh` | Examples of shell scripts with experiment configurations |

## Shell Scripts (`.sh` files)

The `.sh` files contain ready-to-run examples of the federated experiments executed with MLP and LR. The scripts used in the experiments were generated automatically by `create_run_flwr_LR.py` and `create_run_flwr_MLP.py`.

## Generating the Shell Scripts

The two generator scripts read the output of the **centralized training** to set the hyper-parameters for the distributed simulation. This design reflects the experimental setup of the paper: the distributed (federated) simulation leverages knowledge from the centralized training to configure hyper-parameters.

To generate the scripts, run:

```bash
# For Logistic Regression experiments
python create_run_flwr_LR.py

# For MLP experiments
python create_run_flwr_MLP.py
```

Each script will parse the centralized training outputs and produce the corresponding `.sh` files with the appropriate hyper-parameter configurations.

## Evaluating a Trained Model

The `load_model.py` script loads a previously learned federated model and evaluates its performance on the **test dataset** (the same test split used in centralized training), allowing for a direct comparison between centralized and federated performance.

```bash
export PYTHONPATH=$(dirname "$PWD")
python load_model.py
```

## Workflow Overview

```
Centralized Training Output
        │
        ▼
create_run_flwr_LR.py / create_run_flwr_MLP.py
        │
        ▼
   .sh experiment scripts
        │
        ▼
  Federated Simulation (Flower)
        │
        ▼
  load_model.py → Test Evaluation
```
