# Centralized Training Experiments

This folder contains the scripts to reproduce the **centralized training** results for Logistic Regression (LR) and Multilayer Perceptron (MLP) reported in the paper. The outputs of these scripts are also used by the federated experiment generators in `codes/flower/` to configure hyper-parameters.

## Files

| File | Description |
|------|-------------|
| `LR_central.py` | Runs centralized LR experiments across **all datasets and seeds** |
| `MLP_central.py` | Runs centralized MLP experiments for a **single dataset**, specified by its UCI dataset ID |

## Running the Experiments

### Logistic Regression

`LR_central.py` iterates over all datasets and random seeds automatically. Simply run:

```bash
python LR_central.py
```

### MLP

Because MLP training is more computationally expensive, `MLP_central.py` processes one dataset at a time. You must pass the UCI dataset ID as an argument:

```bash
python MLP_central.py <dataset_id>
```

For example, to run on UCI dataset ID=14 (Breast Cancer):

```bash
python MLP_central.py 14
```

To reproduce results for all datasets, run the script once per dataset ID.
