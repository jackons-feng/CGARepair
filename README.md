# CGARepair

Official implementation of CGARepair, a multi-objective local neural network repair framework based on contrastive gradient–activation attribution.

## Overview

CGARepair aims to mitigate defects in pretrained neural networks while preserving their predictive utility. It consists of two main stages:

- **Localization:** Contrastive gradient–activation attribution (GradAct) identifies repair-relevant neurons through task-conditioned contribution comparisons.
- **Repair:** Fast Adaptive Multitask Optimization (FAMO) coordinates defect mitigation and utility preservation within the selected parameter region.

CGARepair supports four categories of neural network defects:

- Adversarial robustness
- Backdoor vulnerabilities
- Safety violations
- Group fairness

## Repository Structure

```text
CGARepair/
├── adv/             # Adversarial robustness repair
├── backdoor/        # Backdoor repair
├── fairness/       # Group fairness repair
├── safety/         # Safety repair
├── rq1/            # Localization and component ablation
├── cgarepair-rq4/  # Optimizer and hyperparameter studies
└── model/          # Original pretrained models
```

## Pretrained Models

The repository provides ten original pretrained neural network models used in the experiments.

| ID | Defect Category | Benchmark | Model File |
|---|---|---|---|
| NN1 | Adversarial | MNIST | `mnist_adv.h5` |
| NN2 | Adversarial | CIFAR-10 | `cifar10_adv.h5` |
| NN3 | Backdoor | Fashion-MNIST | `fashion.h5` |
| NN4 | Backdoor | GTSRB | `gtsrb_resnet.h5` |
| NN5 | Safety | ACAS Xu N1,9 | `a19.h5` |
| NN6 | Safety | ACAS Xu N2,9 | `a29.h5` |
| NN7 | Safety | ACAS Xu N3,3 | `a33.h5` |
| NN8 | Fairness | Bank | `bank.h5` |
| NN9 | Fairness | Adult | `adult.h5` |
| NN10 | Fairness | Default | `default.h5` |

## Experimental Environment

Key software versions recorded in the original experimental environment include:

- Python 3.7.16
- TensorFlow GPU 2.3.0
- PyTorch 1.10.0
- NumPy 1.18.5
- SciPy 1.4.1
- h5py 2.10.0
- pandas 1.3.5
- scikit-learn 0.24.2
- pymoo 0.6.1.1

Additional dependencies may be required by individual baseline implementations.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/jackons-feng/CGARepair.git
cd CGARepair
```

The implementation is organized by defect category. Each task directory contains the corresponding repair and evaluation scripts.

Prepare the benchmark datasets separately and configure the required dataset paths before running the experiments.

The repository includes source code and original pretrained models. Datasets, generated intermediate files, experimental logs, and repaired model checkpoints are not included.

## Experimental Studies

The implementation supports:

- Repair effectiveness evaluation
- Localization effectiveness evaluation
- Component ablation studies
- Optimizer comparisons
- Hyperparameter sensitivity analysis
- Runtime efficiency evaluation

The main repair experiments use five random seeds: 42, 123, 256, 1024, and 2026.

## Citation

Citation information will be updated upon publication of the associated paper.
