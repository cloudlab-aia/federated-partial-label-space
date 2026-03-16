# Structural Stability in Federated Learning with Partial Label Overlap

This repository contains the code and experimental results associated
with the paper:

**"Structural Stability in Federated Learning with Partial Label Overlap"**

The project studies the impact of different **output-space management
strategies** in federated learning scenarios where clients share only a
subset of classes while also possessing **client-specific private
labels**.

The experiments evaluate how architectural decisions affect:

-   performance on **shared global classes**
-   preservation of **client-specific knowledge**
-   stability of collaborative learning under **semantic fragmentation**

------------------------------------------------------------------------

# Repository structure

     federated-partial-label-space
    │
    ├── scripts
    │   └── federated_label_space_experiments.py
    │
    ├── experiments
    │   ├── scenario1
    │   │   ├── run_XXXXXXXX_method
    │   │   │   └── global/server_per_round_metrics.csv
    │   │   └── ...
    │   │
    │   └── scenario2
    │       ├── run_XXXXXXXX_method
    │       │   └── global/server_per_round_metrics.csv
    │       └── ...
    │
    ├── environment.yml
    ├── LICENSE
    └── README.md

**scripts/**\
Contains the main implementation used to run the federated learning
experiments.

**experiments/**\
Contains the metrics generated during the experimental runs.

------------------------------------------------------------------------

# Experimental setting

The experiments simulate a federated learning system composed of **four
clients (hospitals)** using chest X-ray datasets with partially
overlapping label spaces.

The shared global classes are:

-   COVID
-   NORMAL
-   PNEUMONIA

Some clients additionally contain **private classes**, generating
**partial semantic overlap** between participants.

Two experimental scenarios are evaluated:

### Scenario 1 --- Moderate semantic fragmentation

Each client shares the global classes and may contain **one additional
private class**.

### Scenario 2 --- Increased semantic fragmentation

Additional private classes are introduced in selected clients,
increasing the dimensionality of the union label space and creating
stronger semantic asymmetry.

------------------------------------------------------------------------

# Evaluated strategies

Four label-space management strategies are evaluated:

**Union**\
All classes (shared and private) are integrated into a single global
classifier with semantic alignment across clients.

**Naive Union**\
All classes are integrated into a single classifier but without
enforcing semantic alignment between class indices.

**Unknown**\
Private classes are collapsed into a single auxiliary category
(`UNKNOWN`).

**Isolation**\
The global model is restricted to shared classes, while client-specific
classifiers handle private categories locally.

------------------------------------------------------------------------

# Federated training

Federated training is performed using **FedProx**, an extension of
FedAvg designed to mitigate client drift under heterogeneous data
distributions.

Training configuration:

-   4 clients
-   full client participation
-   1 local epoch per round
-   FedProx proximal parameter: **μ = 0.02**

Two communication regimes are evaluated:

-   **20 federated rounds** (3 random seeds)
-   **60 federated rounds** (single seed)

------------------------------------------------------------------------

# Model architecture

The global model is based on **DenseNet-121**, initialised with **Chest
X-ray pre-trained weights** from **TorchXRayVision**.

Images are:

-   converted to grayscale
-   resized to **224×224**
-   normalised using client-specific statistics

Private classifiers in the Isolation strategy are trained locally using
the global backbone as a feature extractor.

------------------------------------------------------------------------

# Datasets

The experiments use several publicly available chest X-ray datasets:

-   COVID-19 Radiography Database\
-   COVID19+PNEUMONIA+NORMAL Chest X-Ray dataset\
-   Chest X-Ray (Pneumonia, COVID-19, Tuberculosis) dataset\
-   Chest X-Ray Image dataset (Mendeley Data)\
-   CASIA-CXR dataset

Due to size restrictions, the datasets are **not included in this
repository**. Please refer to the original dataset sources listed in the
paper.

------------------------------------------------------------------------

# Installation

Create the environment using:

    conda env create -f environment.yml
    conda activate federated-label-space

------------------------------------------------------------------------

# Running experiments

Experiments can be launched using the main script:

    python scripts/federated_label_space_experiments.py

The script runs the federated simulation and stores metrics in the
`experiments/` directory.

------------------------------------------------------------------------

# Reproducibility

The experiments were executed using fixed random seeds:

    42
    123
    456

This allows the stability of the training process to be evaluated across
independent runs.

------------------------------------------------------------------------

# Citation

If you use this repository in your research, please cite the associated
paper.

------------------------------------------------------------------------

