"""Train and evaluate an AQI image classifier, then create a Kaggle submission."""

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import v2

CLASSES = [
    "a_Good",
    "b_Moderate",
    "c_Unhealthy_for_Sensitive_Groups",
    "d_Unhealthy",
    "e_Very_Unhealthy",
    "f_Severe",
]
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_dir(data_dir: Path) -> Path:
    direct = data_dir / "images"
    if direct.is_dir():
        return direct
    candidates = [path for path in data_dir.rglob("images") if path.is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one images/ directory below {data_dir}, found {len(candidates)}."
        )
    return candidates[0]


def find_csv(data_dir: Path, filename: str) -> Path:
    direct = data_dir / filename
    if direct.is_file():
        return direct
    candidates = list(data_dir.rglob(filename))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one {filename} below {data_dir}, found {len(candidates)}."
        )
    return candidates[0]


class AQIDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        transform: v2.Compose,
        labeled: bool,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / str(row["Filename"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image = self.transform(image)
        if not self.labeled:
            return image
        return image, CLASS_TO_IDX[row["AQI_Class"]]


def make_transforms() -> tuple[v2.Compose, v2.Compose]:
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(8),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    return train_transform, eval_transform


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    sampler: DistributedSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_model(pretrained: bool) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def amp_context(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.train()
    loss_sum = torch.zeros(1, device=device)
    example_count = torch.zeros(1, device=device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(amp_enabled):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach() * labels.size(0)
        example_count += labels.size(0)
    if dist.is_initialized():
        dist.all_reduce(loss_sum)
        dist.all_reduce(example_count)
    return (loss_sum / example_count).item()


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            images, batch_labels = batch
            labels.append(batch_labels.numpy())
        else:
            images = batch
        logits = model(images.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    all_probabilities = np.concatenate(probabilities)
    all_labels = np.concatenate(labels) if labels else None
    return all_probabilities, all_labels


@torch.inference_mode()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    loss_sum = 0.0
    example_count = 0
    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)
        logits = model(images.to(device, non_blocking=True))
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        example_count += labels.size(0)
    return loss_sum / example_count


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(CLASSES))[labels]
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "macro_roc_auc": float(
            roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
        ),
        "mae": float(mean_absolute_error(labels, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
    }


def plot_history(history: list[dict[str, float]], output_dir: Path) -> None:
    epochs = np.arange(1, len(history) + 1)
    for metric, title in [("loss", "Loss"), ("accuracy", "Accuracy"), ("macro_roc_auc", "ROC AUC")]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row[f"train_{metric}"] for row in history], label="Train")
        plt.plot(epochs, [row[f"val_{metric}"] for row in history], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"Training and Validation {title}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"training_{metric}.png", dpi=200)
        plt.close()


def plot_confusion(labels: np.ndarray, probabilities: np.ndarray, name: str, output_dir: Path) -> None:
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(CLASSES)))
    figure, axis = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASSES)
    display.plot(ax=axis, xticks_rotation=45, values_format="d", colorbar=False)
    axis.set_title(f"Confusion Matrix - {name.title()}")
    figure.tight_layout()
    figure.savefig(output_dir / f"confusion_matrix_{name}.png", dpi=200)
    plt.close(figure)


def plot_roc(labels: np.ndarray, probabilities: np.ndarray, output_dir: Path) -> None:
    one_hot = np.eye(len(CLASSES))[labels]
    plt.figure(figsize=(9, 7))
    auc_scores = []
    for index, class_name in enumerate(CLASSES):
        false_positive_rate, true_positive_rate, _ = roc_curve(one_hot[:, index], probabilities[:, index])
        score = roc_auc_score(one_hot[:, index], probabilities[:, index])
        auc_scores.append(score)
        plt.plot(false_positive_rate, true_positive_rate, label=f"{class_name} AUC={score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Validation ROC Curve, Macro AUC={np.mean(auc_scores):.4f}")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_auc_val.png", dpi=200)
    plt.close()


def save_checkpoint(model: nn.Module, path: Path, epoch: int, metrics: dict[str, float]) -> None:
    torch.save(
        {"model_state_dict": unwrap_model(model).state_dict(), "epoch": epoch, "metrics": metrics},
        path,
    )


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def validate_csvs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for name, dataframe in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "Filename" not in dataframe:
            raise ValueError(f"{name}_data.csv must contain Filename.")
    for name, dataframe in [("train", train_df), ("val", val_df)]:
        if "AQI_Class" not in dataframe:
            raise ValueError(f"{name}_data.csv must contain AQI_Class.")
        unknown = set(dataframe["AQI_Class"]) - set(CLASSES)
        if unknown:
            raise ValueError(f"{name}_data.csv contains unknown classes: {sorted(unknown)}")
    if "AQI_Class" in test_df:
        raise ValueError("test_data.csv must not contain AQI_Class; hidden test labels cannot be used.")


def write_submission(
    probabilities: np.ndarray, test_df: pd.DataFrame, sample_path: Path, output_dir: Path
) -> None:
    submission = pd.DataFrame(probabilities, columns=CLASSES)
    submission.insert(0, "Filename", test_df["Filename"].to_numpy())
    sample_columns = pd.read_csv(sample_path, nrows=0).columns.tolist()
    if set(sample_columns) != set(submission.columns):
        raise ValueError(f"Submission columns do not match sample_submission.csv: {sample_columns}")
    submission = submission[sample_columns]
    if not np.allclose(submission[CLASSES].sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Submission probabilities do not sum to one.")
    submission.to_csv(output_dir / "submission.csv", index=False)


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    image_dir = find_image_dir(args.data_dir)
    train_df = pd.read_csv(find_csv(args.data_dir, "train_data.csv"))
    val_df = pd.read_csv(find_csv(args.data_dir, "val_data.csv"))
    test_df = pd.read_csv(find_csv(args.data_dir, "test_data.csv"))
    sample_path = find_csv(args.data_dir, "sample_submission.csv")
    validate_csvs(train_df, val_df, test_df)

    train_transform, eval_transform = make_transforms()
    train_dataset = AQIDataset(train_df, image_dir, train_transform, labeled=True)
    train_eval_dataset = AQIDataset(train_df, image_dir, eval_transform, labeled=True)
    val_dataset = AQIDataset(val_df, image_dir, eval_transform, labeled=True)
    test_dataset = AQIDataset(test_df, image_dir, eval_transform, labeled=False)
    sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True, sampler=sampler)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    best_path = args.model_dir / "final_model.pt"
    if not args.predict_only:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(CLASSES)),
            y=train_df["AQI_Class"].map(CLASS_TO_IDX).to_numpy(),
        )
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        history: list[dict[str, float]] = []
        best_auc = -np.inf
        stale_epochs = 0
        for epoch in range(1, args.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp_enabled)
            should_stop = torch.zeros(1, dtype=torch.int32, device=device)
            if is_main(rank):
                train_eval_loader = make_loader(train_eval_dataset, args.batch_size, args.num_workers)
                val_loader = make_loader(val_dataset, args.batch_size, args.num_workers)
                eval_model = unwrap_model(model)
                train_probabilities, train_labels = predict(eval_model, train_eval_loader, device)
                val_probabilities, val_labels = predict(eval_model, val_loader, device)
                train_metrics = calculate_metrics(train_labels, train_probabilities)
                val_metrics = calculate_metrics(val_labels, val_probabilities)
                val_loss = evaluate_loss(eval_model, val_loader, criterion, device)
                row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                row.update({f"train_{key}": value for key, value in train_metrics.items()})
                row.update({f"val_{key}": value for key, value in val_metrics.items()})
                history.append(row)
                print(json.dumps(row, indent=2))
                if val_metrics["macro_roc_auc"] > best_auc:
                    best_auc = val_metrics["macro_roc_auc"]
                    stale_epochs = 0
                    save_checkpoint(model, best_path, epoch, val_metrics)
                else:
                    stale_epochs += 1
                should_stop.fill_(int(stale_epochs >= args.patience))
            if dist.is_initialized():
                dist.broadcast(should_stop, src=0)
                dist.barrier()
            if should_stop.item():
                break
        if is_main(rank):
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
            plot_history(history, args.output_dir)

    if is_main(rank):
        checkpoint_path = args.checkpoint if args.predict_only and args.checkpoint else best_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError("No checkpoint available. Train first or pass --checkpoint.")
        load_checkpoint(unwrap_model(model), checkpoint_path, device)
        inference_model = unwrap_model(model)
        train_eval_loader = make_loader(train_eval_dataset, args.batch_size, args.num_workers)
        val_loader = make_loader(val_dataset, args.batch_size, args.num_workers)
        test_loader = make_loader(test_dataset, args.batch_size, args.num_workers)
        train_probabilities, train_labels = predict(inference_model, train_eval_loader, device)
        val_probabilities, val_labels = predict(inference_model, val_loader, device)
        test_probabilities, _ = predict(inference_model, test_loader, device)
        metrics = {
            "train": calculate_metrics(train_labels, train_probabilities),
            "val": calculate_metrics(val_labels, val_probabilities),
        }
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        report = classification_report(val_labels, val_probabilities.argmax(axis=1), target_names=CLASSES)
        (args.output_dir / "classification_report_val.txt").write_text(report)
        plot_confusion(train_labels, train_probabilities, "train", args.output_dir)
        plot_confusion(val_labels, val_probabilities, "val", args.output_dir)
        plot_roc(val_labels, val_probabilities, args.output_dir)
        write_submission(test_probabilities, test_df, sample_path, args.output_dir)
        print(json.dumps(metrics, indent=2))
        print(f"Wrote {args.output_dir / 'submission.csv'}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
