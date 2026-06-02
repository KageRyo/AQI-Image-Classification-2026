"""Train a ConvNeXt multi-task model for optional AQI and pollutant bonus predictions."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from main import (
    CLASS_TO_IDX,
    build_model,
    find_csv,
    find_image_dir,
    is_main,
    load_checkpoint,
    make_loader,
    make_transforms,
    seed_everything,
    setup_distributed,
    unwrap_model,
    validate_csvs,
)

TARGETS = ["AQI", "PM2.5", "PM10", "O3", "CO", "SO2", "NO2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--classification-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bonus_multitask"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/bonus_multitask"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classification-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--missing-target-strategy",
        choices=["mask", "mean", "median"],
        default="mask",
        help="Mask missing labels or impute them for an explicit ablation experiment.",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


class TargetNormalizer:
    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        fill_values: np.ndarray,
        missing_target_strategy: str,
    ) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.fill_values = fill_values.astype(np.float32)
        self.missing_target_strategy = missing_target_strategy

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, missing_target_strategy: str) -> "TargetNormalizer":
        values = dataframe[TARGETS].to_numpy(dtype=np.float32)
        transformed = np.log1p(values)
        fill_values = np.nanmedian(values, axis=0)
        if missing_target_strategy == "mean":
            fill_values = np.nanmean(values, axis=0)
        return cls(
            np.nanmean(transformed, axis=0),
            np.nanstd(transformed, axis=0),
            fill_values,
            missing_target_strategy,
        )

    def transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(values)
        transformed = (np.log1p(np.where(mask, values, self.fill_values)) - self.mean) / self.std
        loss_mask = mask if self.missing_target_strategy == "mask" else np.ones_like(mask)
        return transformed.astype(np.float32), loss_mask.astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.maximum(np.expm1(values * self.std + self.mean), 0.0)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fill_values": self.fill_values.tolist(),
            "missing_target_strategy": self.missing_target_strategy,
        }


class BonusDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        transform,
        normalizer: TargetNormalizer | None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        with Image.open(self.image_dir / str(row["Filename"])) as source:
            image = self.transform(source.convert("RGB"))
        if self.normalizer is None:
            return image
        values = row[TARGETS].to_numpy(dtype=np.float32)
        targets, mask = self.normalizer.transform(values)
        return image, CLASS_TO_IDX[row["AQI_Class"]], targets, mask


class BonusConvNeXt(nn.Module):
    def __init__(self, classification_checkpoint: Path, device: torch.device) -> None:
        super().__init__()
        backbone = build_model("convnext_tiny", pretrained=False)
        load_checkpoint(backbone, classification_checkpoint, device)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.norm = backbone.classifier[0]
        self.flatten = backbone.classifier[1]
        self.classifier = backbone.classifier[2]
        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.classifier.in_features, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(TARGETS)),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.flatten(self.norm(self.avgpool(self.features(images))))
        return self.classifier(features), self.regressor(features)


def amp_context(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def masked_regression_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.smooth_l1_loss(predictions, targets, reduction="none")
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    classification_loss_weight: float,
) -> float:
    model.train()
    loss_sum = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)
    for images, labels, targets, mask in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(amp_enabled):
            logits, predictions = model(images)
            regression_loss = masked_regression_loss(predictions, targets, mask)
            classification_loss = nn.functional.cross_entropy(logits, labels)
            loss = regression_loss + classification_loss_weight * classification_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach() * labels.size(0)
        count += labels.size(0)
    if dist.is_initialized():
        dist.all_reduce(loss_sum)
        dist.all_reduce(count)
    return (loss_sum / count).item()


@torch.inference_mode()
def predict(model: nn.Module, loader, device: torch.device, tta: bool) -> np.ndarray:
    model.eval()
    predictions = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        _, values = model(images)
        if tta:
            _, flipped_values = model(torch.flip(images, dims=[3]))
            values = (values + flipped_values) / 2
        predictions.append(values.cpu().numpy())
    return np.concatenate(predictions)


def calculate_metrics(
    dataframe: pd.DataFrame, normalized_predictions: np.ndarray, normalizer: TargetNormalizer
) -> dict[str, dict[str, float]]:
    predictions = normalizer.inverse_transform(normalized_predictions)
    metrics = {}
    for index, target in enumerate(TARGETS):
        values = dataframe[target].to_numpy(dtype=np.float32)
        mask = np.isfinite(values)
        metrics[target] = {
            "mae": float(mean_absolute_error(values[mask], predictions[mask, index])),
            "rmse": float(np.sqrt(mean_squared_error(values[mask], predictions[mask, index]))),
        }
    return metrics


def save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    val_normalized_mae: float,
    normalizer: TargetNormalizer,
) -> None:
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "epoch": epoch,
            "val_normalized_mae": val_normalized_mae,
            "normalizer": normalizer.to_dict(),
            "targets": TARGETS,
        },
        path,
    )


def write_bonus_csvs(test_df: pd.DataFrame, predictions: np.ndarray, output_dir: Path) -> None:
    output = pd.DataFrame(predictions, columns=TARGETS)
    output.insert(0, "Filename", test_df["Filename"].to_numpy())
    output[["Filename", "AQI", "PM2.5"]].to_csv(output_dir / "bonus_aqi_pm25.csv", index=False)
    output[["Filename", *TARGETS]].to_csv(output_dir / "bonus_all_metrics.csv", index=False)


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(find_csv(args.data_dir, "train_data.csv"))
    val_df = pd.read_csv(find_csv(args.data_dir, "val_data.csv"))
    test_df = pd.read_csv(find_csv(args.data_dir, "test_data.csv"))
    validate_csvs(train_df, val_df, test_df)
    normalizer = TargetNormalizer.from_dataframe(train_df, args.missing_target_strategy)
    train_transform, eval_transform = make_transforms(224)
    image_dir = find_image_dir(args.data_dir)
    train_dataset = BonusDataset(train_df, image_dir, train_transform, normalizer)
    val_dataset = BonusDataset(val_df, image_dir, eval_transform, normalizer)
    test_dataset = BonusDataset(test_df, image_dir, eval_transform, None)
    sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True, sampler=sampler)

    model = BonusConvNeXt(args.classification_checkpoint, device).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    unwrapped_model = unwrap_model(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": unwrapped_model.features.parameters(), "lr": args.backbone_learning_rate},
            {"params": unwrapped_model.norm.parameters(), "lr": args.head_learning_rate},
            {"params": unwrapped_model.classifier.parameters(), "lr": args.head_learning_rate},
            {"params": unwrapped_model.regressor.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_path = args.model_dir / "final_model.pt"
    best_mae = np.inf
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            args.classification_loss_weight,
        )
        should_stop = torch.zeros(1, dtype=torch.int32, device=device)
        if is_main(rank):
            val_loader = make_loader(val_dataset, args.batch_size, args.num_workers)
            val_predictions = predict(unwrapped_model, val_loader, device, tta=False)
            val_targets, val_mask = normalizer.transform(val_df[TARGETS].to_numpy(dtype=np.float32))
            val_mae = float(np.abs(val_predictions - val_targets)[val_mask.astype(bool)].mean())
            row = {"epoch": epoch, "train_loss": train_loss, "val_normalized_mae": val_mae}
            history.append(row)
            print(json.dumps(row, indent=2))
            if val_mae < best_mae:
                best_mae = val_mae
                stale_epochs = 0
                save_checkpoint(model, best_path, epoch, val_mae, normalizer)
            else:
                stale_epochs += 1
            should_stop.fill_(int(stale_epochs >= args.patience))
        if dist.is_initialized():
            dist.broadcast(should_stop, src=0)
            dist.barrier()
        if should_stop.item():
            break
        scheduler.step()

    if is_main(rank):
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
        val_loader = make_loader(val_dataset, args.batch_size, args.num_workers)
        test_loader = make_loader(test_dataset, args.batch_size, args.num_workers)
        val_predictions = predict(unwrapped_model, val_loader, device, tta=args.tta)
        test_predictions = predict(unwrapped_model, test_loader, device, tta=args.tta)
        metrics = calculate_metrics(val_df, val_predictions, normalizer)
        output_values = normalizer.inverse_transform(test_predictions)
        np.save(args.output_dir / "test_predictions.npy", output_values)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        write_bonus_csvs(test_df, output_values, args.output_dir)
        print(json.dumps(metrics, indent=2))
        print(f"Wrote {args.output_dir / 'bonus_all_metrics.csv'}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
