# CGARepair

Official implementation of CGARepair: Contrastive Gradient–Activation Attribution for Multiobjective Local Neural Network Repair.

## Overview

CGARepair is a two-stage post-training neural network repair framework that aims to mitigate defects in pretrained neural networks while preserving their predictive utility.

- **Localization:** Contrastive gradient–activation attribution (GradAct) identifies repair-relevant neurons through task-conditioned contribution comparisons and constructs an editable subspace of incoming weights.
- **Repair:** Box-projected Fast Adaptive Multitask Optimization (FAMO) coordinates defect mitigation and utility preservation within the selected parameter region while keeping unselected parameters frozen.

CGARepair supports four categories of neural network defects:

- Adversarial robustness
- Backdoor vulnerabilities
- Safety violations
- Group fairness

The associated paper evaluates CGARepair on nine pretrained neural networks across these four defect categories.

## Repository Structure

```text
CGARepair/
├── adv/             # Adversarial robustness repair
├── backdoor/        # Backdoor repair
├── fairness/        # Group fairness repair
├── safety/          # Safety repair
├── rq1/             # Localization and component ablation
├── cgarepair-rq4/   # Optimizer and hyperparameter studies
└── model/           # Original pretrained models
```

## Pretrained Models

The following nine original pretrained neural network models are used in the experiments reported in the paper.

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

The repository additionally retains the Default model (`default.h5`), which is not included in the nine-model evaluation reported in the paper.

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

The experiments reported in the paper were conducted on an NVIDIA Tesla V100S GPU with 32 GB of memory.

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

## Experimental Protocol

The main repair experiments use the following settings:

| Setting | Configuration |
|---|---|
| Localization method | GradAct |
| Repair optimizer | FAMO |
| Localization seed | 256 |
| Optimization seeds | 42, 123, 256, 1024, 2026 |
| FAMO update steps | 5,000 |
| Checkpoint interval | 20 |
| Local parameter bounds | [-5, 5] |
| Learning rate | 0.001 |
| Task-weight decay coefficient | 0.001 |

The cumulative neuron ablation experiments use five localization seeds: 42, 123, 256, 1024, and 2026.

In the optimizer comparison, FAMO uses 5,000 update steps, while MOEA/D and NSGA-III each use 5,000 candidate evaluations.

Localization, repair optimization, and checkpoint selection are performed using training data. Held-out test data are reserved for final evaluation.

## Experimental Studies

The implementation supports:

- Repair effectiveness evaluation across four defect categories
- Cumulative top-k neuron ablation
- Localization comparisons with AD and IGD
- Localization replacement studies
- Reference-contrast ablation
- Fairness-direction ablation
- Localization runtime evaluation
- Optimizer comparisons
- Update budget and runtime analysis

The experimental evaluation reported in the paper covers nine pretrained neural networks and eight repair methods from existing studies.

## Citation

Citation information will be updated upon publication of the associated paper.
