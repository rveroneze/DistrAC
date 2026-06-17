# Codes

This folder contains all the code necessary to reproduce the experiments presented in the paper, organized into subfolders by experiment type. It also includes a `requirements.txt` for setting up the Python environment.

## Setup

Install all required dependencies with:

```bash
pip install -r requirements.txt
```

## Subfolders

| Folder | Description |
|--------|-------------|
| [`common/`](common/) | Shared utility code used across experiments |
| [`centralized/`](centralized/) | Scripts to reproduce the **centralized** LR and MLP baseline results |
| [`flower/`](flower/) | Scripts to reproduce the **federated** LR and MLP experiments using the [Flower (flwr)](https://flower.ai/) framework |
| [`PM/`](PM/) | Scripts to reproduce the **associative classification** experiments (centralized and distributed) |

## Experiment Overview

The table below summarizes which subfolder to use depending on the method and training setting:

| Method | Setting | Folder |
|--------|---------|--------|
| Logistic Regression (LR) | Centralized | `centralized/` |
| MLP | Centralized | `centralized/` |
| Logistic Regression (LR) | Federated | `flower/` |
| MLP | Federated | `flower/` |
| Associative Classifiers (AC) | Centralized & Distributed | `PM/` |
| DAC | Centralized & Distributed | `PM/` |

