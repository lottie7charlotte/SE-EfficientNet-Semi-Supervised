# SE-EfficientNetV2-S with Semi-Supervised Learning

Official PyTorch implementation for the paper: *"Research on Remote Sensing Image Scene Classification Based on Improved EfficientNet and Semi-Supervised Learning"* (Submitted to *Sensors*, 2026).

## 🗂️ Dataset Preparation
We strictly utilized a 10-class subset of the NWPU-RESISC45 benchmark. To ensure absolute transparency and reproducibility, the dataset is physically partitioned into:
- **Train (Labeled):** 3,920 images (392 per class)
- **Unlabeled Pool:** 980 images (98 per class)
- **Validation:** 700 images (70 per class)
- **Test:** 1,400 images (140 per class)

Run `build_dataset.py` on the raw NWPU-RESISC45 dataset to reproduce this exact mathematical split (Seed=2026).

##  Quick Start
### 1. Supervised Baseline Training (Phase 1)
- Run `train.py` for the proposed SE-EfficientNetV2-S base model (fully unfrozen).
- Run `train_no_se.py` for the ablation baseline without the SE block.

### 2. Semi-Supervised Fine-Tuning (Phase 2)
- Run `pseudo_finetune.py` to extract high-confidence pseudo-labels (τ=0.90) and execute full-parameter self-training.

### 3. Evaluation & Reporting
- Run `evaluate_test.py` to evaluate all checkpoints on the independent 1,400-sample test set (Peak Accuracy: 99.57%).
- Run `generate_report.py` to output the exact per-class metrics and generate the confusion matrix.

##  Comparative Baselines & Reviewer Proofs
To ensure absolute rigor and address reviewer inquiries, the following experimental scripts are included:
- `compare_resnet.py` & `pseudo_finetune_resnet.py`: ResNet50 fair baseline comparison.
- `compare_mobilenet.py`: MobileNetV3-Small fair baseline comparison.
- `run_variance.py`: Automatically runs 3 different random seeds to calculate the Mean ± Std for per-class F1-scores.
- `sensitivity_test.py`: Ablation study for pseudo-labeling confidence thresholds (τ=0.80, 0.90, 0.95).
- `clean_now.py`: Utility script to physically purge pseudo-labeled images and prevent data leakage.

##  Hardware Notes
The training scripts utilize gradient accumulation (`accumulation_steps = 2` with `batch_size = 4`) to ensure the model trains perfectly on resource-constrained GPUs (e.g., 4GB VRAM) without Out-Of-Memory (OOM) errors. All experiments are fixed with `set_seed(2026)` via `utils.py` for strict reproducibility.
