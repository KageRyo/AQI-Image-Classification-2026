"""Refit a selected checkpoint on all public labeled data and predict the test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from main import (
    AQIDataset,
    CLASSES,
    CLASS_TO_IDX,
    build_model,
    default_image_size,
    find_csv,
    find_image_dir,
    is_main,
    load_checkpoint,
    make_loader,
    make_transforms,
    predict,
    read_checkpoint_metadata,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    train_one_epoch,
    unwrap_model,
    validate_csvs,
    write_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48, help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Classifier learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    metadata = read_checkpoint_metadata(args.checkpoint)
    model_name = metadata["model_name"]
    image_size = metadata.get("image_size", default_image_size(model_name))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    image_dir = find_image_dir(args.data_dir)
    train_df = pd.read_csv(find_csv(args.data_dir, "train_data.csv"))
    val_df = pd.read_csv(find_csv(args.data_dir, "val_data.csv"))
    test_df = pd.read_csv(find_csv(args.data_dir, "test_data.csv"))
    sample_path = find_csv(args.data_dir, "sample_submission.csv")
    validate_csvs(train_df, val_df, test_df)
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    train_transform, eval_transform = make_transforms(image_size)
    full_dataset = AQIDataset(full_df, image_dir, train_transform, labeled=True)
    test_dataset = AQIDataset(test_df, image_dir, eval_transform, labeled=False)
    sampler = DistributedSampler(full_dataset, shuffle=True) if world_size > 1 else None
    train_loader = make_loader(full_dataset, args.batch_size, args.num_workers, shuffle=True, sampler=sampler)

    model = build_model(model_name, pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    unwrapped_model = unwrap_model(model)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASSES)),
        y=full_df["AQI_Class"].map(CLASS_TO_IDX).to_numpy(),
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": unwrapped_model.features.parameters(), "lr": args.backbone_learning_rate},
            {"params": unwrapped_model.classifier.parameters(), "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    final_path = args.model_dir / "final_model.pt"
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp_enabled)
        if is_main(rank):
            row = {"epoch": epoch, "train_loss": loss}
            history.append(row)
            print(json.dumps(row, indent=2))
            save_checkpoint(model, final_path, epoch, row, model_name, image_size)
        if dist.is_initialized():
            dist.barrier()

    if is_main(rank):
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        test_loader = make_loader(test_dataset, args.batch_size, args.num_workers)
        test_probabilities, _ = predict(unwrapped_model, test_loader, device, tta=args.tta)
        np.save(args.output_dir / "test_probabilities.npy", test_probabilities)
        write_submission(test_probabilities, test_df, sample_path, args.output_dir)
        print(f"Wrote {args.output_dir / 'submission.csv'}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

