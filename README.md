# Remote Sensing Scene Classification with SE-EfficientNet and Semi-Supervised Learning

Official PyTorch implementation for the paper: **"Research on Remote Sensing Image Scene Classification Based on Improved EfficientNet and Semi-Supervised Learning"** published in *Sensors* (MDPI).

This repository contains the complete pipeline for a lightweight, high-performance remote sensing scene classification framework. It integrates an improved EfficientNetV2-S backbone with a Squeeze-and-Excitation (SE) channel attention module operating on pre-pooling $7 \times 7$ spatial feature maps, combined with a dual-phase hybrid semi-supervised pseudo-labeling optimization strategy.

---

## 1. Project Architecture & Repository Structure

```text
.
├── checkpoints/               # Directory for saved model weights (.pth)
├── checkpoints_variance/      # Model weights for multi-seed variance analysis
├── data/                      # Data root directory
│   ├── NWPU-RESISC45/         # Raw source benchmark dataset (10 target classes)
│   └── RS_Classification/     # Processed and partitioned pure dataset splits
├── build_dataset.py           # Strictly seeded dataset partition and splitting script
├── data_loader.py             # Preprocessing transforms, augmentations, and PyTorch DataLoaders
├── model.py                   # Network definition for proposed SE-EfficientNetV2-S
├── model_no_se.py             # Network definition for No-SE baseline model
├── train.py                   # Fully-unfrozen pure supervised primary training script
├── train_no_se.py             # Primary training script for the No-SE baseline configuration
├── compare_mobilenet.py       # Training and evaluation baseline for MobileNetV3-Small
├── compare_resnet.py          # Training and evaluation baseline for SEResNet50
├── pseudo_finetune.py         # Phase 2 semi-supervised pseudo-label fine-tuning script
├── pseudo_finetune_no_se.py   # Semi-supervised fine-tuning for the No-SE ablation core
├── pseudo_finetune_resnet.py  # Semi-supervised fine-tuning for the ResNet50 baseline
├── evaluate_test.py           # Independent test dataset accuracy verification engine
├── generate_report.py         # Classification report generator and confusion matrix visualizer
├── run_variance.py            # Multi-seed variance analyzer (generates Table 9 Mean ± Std)
├── sensitivity_test.py        # Confidence threshold (tau) sensitivity analysis script
├── utils.py                   # Deterministic execution utilities (seed locking)
├── requirements.txt           # Python software dependency manifest
└── README.md                  # Comprehensive repository documentation
```

---

## 2. Environment Setup & Prerequisites

The framework is optimized for maximum resource efficiency, specifically tailored to execute full-parameter tuning on hardware with strict memory constraints (e.g., 4GB VRAM GPUs) via gradient accumulation techniques.

Install the exact package dependencies using `pip`:

```bash
pip install -r requirements.txt
```

*Note: PyTorch version $\ge$ 2.0.0 and Torchvision version $\ge$ 0.15.0 are recommended for full compatibility with native weight loading parameters.*

---

## 3. Dataset Engineering & Splitting

The experimental setup is evaluated on a 10-class geographic land-use subset extracted from the public **NWPU-RESISC45** dataset. The target classes include: `airplane`, `airport`, `beach`, `bridge`, `forest`, `freeway`, `harbor`, `industrial_area`, `parking_lot`, and `stadium`.
### Dataset & Pre-trained Weights Download
To ensure strict reproducibility as described in our manuscript, the exact 10-class dataset splits (7,000 images) and the pre-trained `.pth` model weights have been archived on Zenodo. 
You can directly download the `RS_Classification.zip` and weight files from our official repository:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20995485.svg)](https://doi.org/10.5281/zenodo.20995485)

### Steps to prepare the data:
1. Download the benchmark dataset from the Official NWPU-RESISC45 Repository.
2. Decompress and align the folders under the repository as follows:
   ```text
   ./data/NWPU-RESISC45/
   ├── airplane/
   ├── airport/
   └── ... (all 10 selected categories)
   ```
3. Execute the seeded dataset engineering script. This will physically isolate the data into clean training, validation, testing, and unlabeled partitions using an un-skewed deterministic random partition (Seed: 2026):
   ```bash
   python build_dataset.py
   ```

---

## 4. Operational Optimization Workflow

The optimization workflow is split into two separate phases to bridge the gap between ImageNet pre-training and specialized remote sensing domains without causing structural underfitting.

### Phase 1: Pure Supervised Primary Training
To train the main proposed SE-EfficientNetV2-S model under strictly fair, fully-unfrozen conditions with an effective batch size simulated via gradient accumulation:
```bash
python train.py
```
*To populate the backbone comparison matrix, you can also execute the respective baseline variants:*
```bash
python train_no_se.py
python compare_resnet.py
python compare_mobilenet.py
```

### Phase 2: Semi-Supervised Pseudo-Label Fine-Tuning
After primary supervised convergence, the optimal base checkpoint is loaded to act as a high-confidence pseudo-label filter over the unannotated partition pool using a strict confidence threshold of $\tau = 0.90$:
```bash
python pseudo_finetune.py
```
*For component ablation analysis, fine-tune the No-SE or ResNet50 baseline equivalents using:*
```bash
python pseudo_finetune_no_se.py
python pseudo_finetune_resnet.py
```

---

## 5. Evaluation, Reporting, & Reproducibility

To ensure strict compliance with scientific reproducibility standards, multiple validation scripts are provided to compute and output final metrics directly from independent testing streams (1,400 balanced frames).

### Main Performance Metrics
To calculate the final standalone Test Accuracy across all evaluated models simultaneously:
```bash
python evaluate_test.py
```

### Detailed Diagnostic Evaluation
To generate the comprehensive precision-recall-F1 classification matrix and plot the high-resolution publication-ready confusion matrix heatmap (`New_confusion_matrix.png`):
```bash
python generate_report.py
```

### Statistical Stability & Multi-Seed Analysis
To verify stable convergence and reproduce the exact mean and standard deviation boundaries documented in **Table 9** across three independent random seeds (1024, 2026, 3038):
```bash
python run_variance.py
```

### Hyperparameter Sensitivity Testing
To run a complete hyperparameter sweep across alternative confidence boundaries ($\tau = 0.80$ and $\tau = 0.95$) to evaluate pseudo-labeling filtration threshold fragility:
```bash
python sensitivity_test.py
```

---

## 6. Main Empirical Benchmarks Summary

For quick reference during manuscript proofreading, the definitive empirical results achieved by this codebase under fully-unfrozen conditions are summarized below:

* **Proposed Supervised SE-EfficientNetV2-S:** Achieve **98.71%** Independent Test Accuracy, outperforming ResNet50 (**98.50%**) while using 15% fewer network parameters (20.4M vs. 24.1M).
* **Semi-Supervised Trajectory:** Pseudo-labeling raises internal validation accuracy to a peak of **99.57%**, but multi-seed independent test evaluation reveals a minor over-fitting trade-off (**98.43%**), establishing the pure supervised variant as the most robust model for deployment.

---

## Citation

If you find this codebase, data splits, or experimental findings helpful in your academic research, please consider citing our original manuscript:

```bibtex
@article{liao2026research,
  title={Research on Remote Sensing Image Scene Classification Based on Improved EfficientNet and Semi-Supervised Learning},
  author={Liao, Liting and Yang, Haoyuan and Peng, Jun and Jin, Runqiu},
  journal={Sensors},
  volume={26},
  year={2026},
  publisher={MDPI}
}
```
