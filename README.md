# AQI Image Classification 2026

PyTorch baseline for the 2026 Deep Learning final exam Kaggle competition.
The model classifies each air-pollution image into one of six AQI classes and
writes the six probabilities required by Kaggle.

## Environment

The existing `dl-class-ryo` Conda environment already contains PyTorch,
Torchvision, Pandas, NumPy, Matplotlib, and scikit-learn. Use it for this
project:

```bash
conda activate dl-class-ryo
```

Jupyter is not required. The complete workflow runs from `main.py`.

Install this project in editable mode if the environment needs any missing
dependencies:

```bash
python -m pip install -e .
```

## Dataset

Download only the public competition files and place them under `data/`:

```text
data/
├── images/
├── train_data.csv
├── val_data.csv
├── test_data.csv
└── sample_submission.csv
```

`test_data.csv` must contain filenames only. Do not use reconstructed or
leaked test labels.

After joining the competition and configuring Kaggle authentication, download
and extract the public files with:

```bash
python download_data.py
```

## Train

Use one GPU for an initial run:

```bash
python main.py --data-dir data --epochs 3
```

Use both local RTX 4090 GPUs for the full experiment:

```bash
torchrun --standalone --nproc_per_node=2 main.py \
  --data-dir data \
  --epochs 30 \
  --batch-size 64
```

`--batch-size` is per GPU, so the two-GPU effective batch size is `128`.
Reduce it if GPU memory is insufficient.

For the stronger EfficientNet-B2 experiment, use its default 288-pixel input
size and horizontal-flip test-time augmentation:

```bash
torchrun --standalone --nproc_per_node=2 main.py \
  --data-dir data \
  --output-dir outputs/efficientnet_b2 \
  --model-dir models/efficientnet_b2 \
  --model-name efficientnet_b2 \
  --epochs 30 \
  --batch-size 48 \
  --tta
```

EfficientNet-B3 is available as a more diverse 300-pixel ensemble candidate:

```bash
torchrun --standalone --nproc_per_node=2 main.py \
  --data-dir data \
  --output-dir outputs/efficientnet_b3 \
  --model-dir models/efficientnet_b3 \
  --model-name efficientnet_b3 \
  --epochs 30 \
  --batch-size 32 \
  --tta
```

ConvNeXt Tiny provides a second architecture family for a more diverse
ensemble candidate:

```bash
torchrun --standalone --nproc_per_node=2 main.py \
  --data-dir data \
  --output-dir outputs/convnext_tiny \
  --model-dir models/convnext_tiny \
  --model-name convnext_tiny \
  --epochs 30 \
  --batch-size 32 \
  --tta
```

## Outputs

The best validation ROC AUC checkpoint is saved to:

```text
models/final_model.pt
```

The script writes these generated files under `outputs/`:

```text
submission.csv
history.json
metrics.json
classification_report_val.txt
training_loss.png
training_accuracy.png
training_macro_roc_auc.png
confusion_matrix_train.png
confusion_matrix_val.png
roc_auc_val.png
```

Generated data, models, and outputs are ignored by Git.

## Ensemble

Training writes reusable train, validation, and test probability arrays. Blend
two experiments and select the best validation ROC AUC weight with:

```bash
python ensemble.py \
  --first-output-dir outputs/efficientnet_b2 \
  --second-output-dir outputs/efficientnet_b2_seed_123 \
  --output-dir outputs/ensemble_b2
```

Blend three diverse experiments with:

```bash
python ensemble_three.py \
  --first-output-dir outputs/efficientnet_b2 \
  --second-output-dir outputs/efficientnet_b2_seed_123 \
  --third-output-dir outputs/efficientnet_b3 \
  --output-dir outputs/ensemble_b2_b3
```

## Full-Data Refit

After model selection, legally reuse all public `train_data.csv` and
`val_data.csv` labels for a low-learning-rate final refit:

```bash
torchrun --standalone --nproc_per_node=2 refit.py \
  --data-dir data \
  --checkpoint models/efficientnet_b3/final_model.pt \
  --output-dir outputs/efficientnet_b3_refit \
  --model-dir models/efficientnet_b3_refit \
  --epochs 3 \
  --batch-size 32 \
  --tta
```

Refit removes the held-out validation split. Reuse validation-selected weights
instead of tuning against hidden test results:

```bash
python blend_fixed.py \
  --output-dirs outputs/efficientnet_b2_refit outputs/efficientnet_b3_refit \
  --weights 45 55 \
  --output-dir outputs/ensemble_refit
```
