# SE-EfficientNet-Semi-Supervised
Official PyTorch implementation for: "Research on Remote Sensing Image Scene Classification Based on Improved EfficientNet and Semi-Supervised Learning"
## Dataset Preparation
We strictly utilized a 10-class subset of the NWPU-RESISC45 benchmark. To ensure absolute transparency and reproducibility, the dataset is split perfectly into:
- **Train (Labeled):** 3,920 images (392 per class)
- **Unlabeled Pool:** 980 images (98 per class)
- **Validation:** 700 images (70 per class)
- **Test:** 1,400 images (140 per class)

Run `split_dataset.py` on the raw NWPU-RESISC45 dataset to reproduce this exact mathematical split.

## Quick Start
1. **Supervised Baseline:** Run `train_base.py` (Unfrozen, Batch Size 8 via accumulation).
2. **Semi-Supervised Fine-Tuning:** Run `finetune.py` to extract high-confidence pseudo-labels (τ=0.90) and execute full-parameter self-training.
3. **Evaluation:** Run `test_all.py` to test on the independent 1,400-sample test set (Peak Accuracy: 99.57%).

## Hardware Notes
The training scripts utilize gradient accumulation (`accumulation_steps = 2` with `batch_size = 4`) to ensure the model trains perfectly on resource-constrained GPUs (e.g., 4GB VRAM).
