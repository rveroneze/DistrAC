# Associative Classification Experiments

This folder contains the code for the experiments with **associative classification (AC)** algorithms, covering both centralized and distributed training settings. The pipeline relies on the [RIn-Close algorithm](https://github.com/rveroneze/rinclose/tree/mminsup) for closed itemset enumeration, followed by rule-based classification steps.

## Files

### Core Components

| File | Description |
|------|-------------|
| `RInClose_mminsup` | Compiled binary of the RIn-Close algorithm, used to enumerate closed itemsets (rules). Source available at [rveroneze/rinclose (mminsup branch)](https://github.com/rveroneze/rinclose/tree/mminsup) |
| `utils_PM.py` | Utility methods for Pattern Mining, specifically for associative classification |
| `base.py` | Base class for generic associative classification algorithms — includes rule ranking, rule pruning, and prediction schemes |
| `AC_consolidation.py` | Class for consolidating local models created at each client into a single global model |
| `DAC2017.py` | Implementation of the DAC algorithm |

### Experiment Scripts

| File | Arguments | Description |
|------|-----------|-------------|
| `create_data_run_rinclose.py` | `dataset_id  n_sites  seed` | Prepares input files and runs RIn-Close to enumerate rules (patterns) |
| `ACs_central_run.py` | `dataset_id  seed` | Runs associative classifiers with **centralized** training |
| `ACs_dist_create_local_clfs.py` | `dataset_id  n_sites  seed` | Creates **local** associative classifiers (distributed training) |
| `ACs_dist_create_global_clf.py` | `dataset_id  n_sites  seed` | Consolidates local classifiers into a **global** model |
| `ACs_dist_predictions.py` | `dataset_id  n_sites` | Reads global classifiers, runs predictions, and saves results (loops over all seeds) |
| `DAC_run.py` | `dataset_id  n_sites` | Runs experiments with the DAC algorithm; `n_sites=1` → centralized, `n_sites>1` → distributed |

## Running the Experiments

All examples below use **UCI dataset ID 14** (Breast Cancer), **2 sites (clients)**, and **seed 42**.

### DAC Algorithm

```bash
# Centralized (n_sites=1) or distributed (n_sites>1)
python DAC_run.py 14 2
```

### Associative Classifiers — Centralized Training

```bash
# Step 1: Enumerate rules
python create_data_run_rinclose.py 14 1 42

# Step 2: Train and evaluate
python ACs_central_run.py 14 42
```

### Associative Classifiers — Distributed Training

```bash
# Step 1: Enumerate rules
python create_data_run_rinclose.py 14 2 42

# Step 2: Create local classifiers at each client
python ACs_dist_create_local_clfs.py 14 2 42

# Step 3: Consolidate local models into a global classifier
python ACs_dist_create_global_clf.py 14 2 42

# Step 4: Run predictions with the global classifier (loops over all seeds)
python ACs_dist_predictions.py 14 2
```

## Workflow Overview

```
                         ┌─────────────────────────────────┐
                         │   create_data_run_rinclose.py   │
                         │  (enumerate rules via RIn-Close) │
                         └────────────┬────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
              ▼                                               ▼
   CENTRALIZED TRAINING                          DISTRIBUTED TRAINING
              │                                               │
   ACs_central_run.py                   ACs_dist_create_local_clfs.py
                                                              │
                                        ACs_dist_create_global_clf.py
                                                              │
                                           ACs_dist_predictions.py
```

> **Note 1:** The output paths for saved files are configured inside each script — adjust them to match your directory structure before running.

> **Note 2:** The files ending in **_noniid** contain the scripts used for the preliminary experiments on non-IID data partitioning with quantity skew.
