# AQI Image Classification 2026

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Kaggle Public ROC AUC](https://img.shields.io/badge/Kaggle_Public_ROC_AUC-1.00000-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/2026-dl-final-exam-india-nepal-aqi-classification)

PyTorch baseline for the 2026 Deep Learning final exam Kaggle competition.
The model classifies each air-pollution image into one of six AQI classes and
writes the six probabilities required by Kaggle.

## Results

The final submission is a fixed-weight ensemble of EfficientNet-B2,
EfficientNet-B3, and ConvNeXt Tiny checkpoints. The weights were selected using
the public validation split before the selected checkpoints were refit on all
public labeled data.

| Metric | Score |
|---|---:|
| Validation ensemble ROC AUC | `0.99993954` |
| Kaggle Public ROC AUC | `1.00000` |

The Kaggle Private score remains unavailable until the competition closes.

## Environment

Create and activate a Python 3.10+ Conda environment:

```bash
conda create --name <environment-name> python=3.10
conda activate <environment-name>
```

Jupyter is not required. The complete workflow runs from `main.py`.

Install the project and its dependencies in editable mode:

```bash
python -m pip install -e .
```

Install development dependencies and run the CPU unit tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
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

For the validation-selected EfficientNet and ConvNeXt ensemble, refit all four
checkpoints and reuse the fixed validation weights:

```bash
python blend_fixed.py \
  --output-dirs \
    outputs/efficientnet_b2_refit \
    outputs/efficientnet_b2_seed_123_refit \
    outputs/efficientnet_b3_refit \
    outputs/convnext_tiny_refit \
  --weights 6 39 5 50 \
  --output-dir outputs/ensemble_refit_b2_b3_convnext
```

## Optional Bonus Predictions

The optional multi-task workflow predicts `AQI`, `PM2.5`, `PM10`, `O3`, `CO`,
`SO2`, and `NO2` without changing the Kaggle classification submission:

```bash
torchrun --standalone --nproc_per_node=2 bonus_multitask.py \
  --data-dir data \
  --classification-checkpoint models/convnext_tiny/final_model.pt \
  --output-dir outputs/bonus_multitask \
  --model-dir models/bonus_multitask \
  --epochs 20 \
  --batch-size 8 \
  --tta
```

Some pollutant labels are missing in the public training data. The default
`--missing-target-strategy mask` excludes only those missing labels from the
regression loss. `mean` and `median` are available for explicit ablation
experiments, but they introduce imputed target values.

## References

This project uses only the public files distributed through the course Kaggle
competition. The original dataset repository is cited for dataset provenance
and documentation:

- [2026 DL Final Exam - India & Nepal AQI Classification: Citation](https://www.kaggle.com/competitions/2026-dl-final-exam-india-nepal-aqi-classification/overview/citation)
- [Air Pollution Image Dataset From India and Nepal: GitHub Repository](https://github.com/ICCC-Platform/Air-Pollution-Image-Dataset-From-India-and-Nepal)
- Rouniyar, A., Utomo, S., A, J., & Hsiung, P.-A. (2023). *Air Pollution
  Image Dataset from India and Nepal* [Data set]. Kaggle.
  [https://doi.org/10.34740/KAGGLE/DS/3152196](https://doi.org/10.34740/KAGGLE/DS/3152196)
