# DistrAC — Distributed Associative Classification

This repository contains the code, datasets, and results associated with the paper:

> **Towards associative classification in distributed environments**  
> Rosana Veroneze, Xavier Lessage, Jean Vanderdonckt  
> *To be presented at [ECML PKDD 2026](https://ecmlpkdd.org/2026/) — September 7–11, Naples, Italy*

## Abstract

DistrAC (**Distr**ibuted **A**ssociative **C**lassification) investigates associative classification algorithms in distributed environments, comparing centralized and distributed training settings across multiple UCI benchmark datasets. The paper also benchmarks against standard machine learning methods (Logistic Regression and MLP) trained both in centralized and federated settings using the Flower framework.

## Repository Structure

| Folder | Description |
|--------|-------------|
| [`codes/`](codes/) | All experiment scripts — see `codes/README.md` for setup and usage |
| [`datasets/`](datasets/) | Datasets used in the experiments, all sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) |
| [`figures/`](figures/) | Additional result figures not included in the paper |
| [`results/`](results/) | Result files used in the paper, plus extra results omitted due to space constraints |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rveroneze/DistrAC.git
cd DistrAC
```

### 2. Install dependencies

```bash
pip install -r codes/requirements.txt
```

### 3. Run experiments

Refer to the `README.md` inside each subfolder of `codes/` for detailed instructions:

- [`codes/centralized/`](codes/centralized/) — Centralized LR and MLP baselines
- [`codes/flower/`](codes/flower/) — Federated LR and MLP experiments (Flower framework)
- [`codes/PM/`](codes/PM/) — Associative classification experiments (centralized and distributed)

## Datasets

All datasets are from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) and are provided in the `datasets/` folder for reproducibility.

## Extra Results

The `results/` folder includes all result files referenced in the paper, as well as supplementary experiments not included due to space constraints, such as:

- Alternative prediction schemes (beyond rule lists)
- Preliminary non-IID data partition experiments with quantity skew, varying the α parameter of a Dirichlet distribution

## Citation

If you use this code or results in your work, please cite:

```bibtex
@inproceedings{veroneze2026distractac,
  title     = {Towards associative classification in distributed environments},
  author    = {Veroneze, Rosana and Lessage, Xavier and Vanderdonckt, Jean},
  booktitle = {Proceedings of ECML PKDD 2026},
  year      = {2026}
}
```
