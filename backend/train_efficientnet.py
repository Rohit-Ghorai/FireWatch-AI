"""
EfficientNet-B0 Fine-tuning Script for Fire / No-Fire Classification.

Usage
-----
  python train_efficientnet.py \\
      --dataset  <path/to/dataset>   # must have fire/ and no_fire/ subdirs
      --epochs   20                  # default 20
      --batch-size 32                # default 32
      --lr       1e-4                # default 1e-4
      --output   efficientnet_fire_best.pth

Dataset layout expected
-----------------------
  dataset/
    fire/
      img001.jpg
      img002.jpg
      ...
    no_fire/
      img001.jpg
      img002.jpg
      ...

The script will:
  1. Validate the dataset (report corrupt / missing images, class distribution).
  2. Split 70 % train / 15 % val / 15 % test using a fixed seed (reproducible).
  3. Apply class-imbalance weighting (weighted cross-entropy loss).
  4. Fine-tune EfficientNet-B0 — freeze backbone for first half of epochs,
     then unfreeze all layers for the second half.
  5. Use validation accuracy for model selection (never test data).
  6. Save the best checkpoint and a config JSON.
  7. Print a training history table.

Outputs
-------
  efficientnet_fire_best.pth    — best model weights (val-accuracy criterion)
  efficientnet_config.json      — class mapping + architecture info
  training_history.json         — per-epoch loss & accuracy
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image

SEED = 42
# Canonical class mapping: class index 0 = no_fire, class index 1 = fire
CLASSES = ["no_fire", "fire"]
CLASS_TO_IDX = {"no_fire": 0, "fire": 1}
IMG_SIZE = 224


class CanonicalImageFolder(ImageFolder):
    """ImageFolder that enforces canonical class indices (0=no_fire, 1=fire)."""
    def find_classes(self, directory: str):
        return CLASSES, CLASS_TO_IDX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_dataset(root: Path) -> dict:
    """Check dataset integrity and return per-class image counts."""
    report = {}
    for cls in CLASSES:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            print(f"[Dataset] ERROR — missing class folder: {cls_dir}")
            sys.exit(1)

        valid, corrupt = 0, 0
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            try:
                with Image.open(img_path) as im:
                    im.verify()
                valid += 1
            except Exception:
                print(f"[Dataset] Corrupt image (skipped): {img_path}")
                corrupt += 1

        if valid == 0:
            print(f"[Dataset] ERROR — class '{cls}' has 0 valid images. "
                  "Add images before training.")
            sys.exit(1)

        report[cls] = {"valid": valid, "corrupt": corrupt}
        print(f"[Dataset]   {cls:10s}: {valid:5d} valid, {corrupt} corrupt")

    return report


def stratified_split(dataset, val_frac=0.15, test_frac=0.15, seed=SEED):
    """
    Returns (train_indices, val_indices, test_indices) using per-class
    stratified sampling with a fixed seed.
    """
    rng = random.Random(seed)
    cls_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        cls_to_indices.setdefault(label, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in cls_to_indices.items():
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    return train_idx, val_idx, test_idx


def make_sampler(dataset, indices):
    """Weighted sampler to handle class imbalance in training."""
    labels = [dataset.targets[i] for i in indices]
    class_counts = [labels.count(c) for c in range(len(CLASSES))]
    class_weights = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(augment: bool):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 for fire detection")
    parser.add_argument("--dataset",    default="../dataset",           help="Root dataset directory")
    parser.add_argument("--split-file",  default=None,                   help="Pre-generated dataset split JSON file")
    parser.add_argument("--epochs",     type=int,   default=20,         help="Total training epochs")
    parser.add_argument("--batch-size", type=int,   default=32,         help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4,       help="Learning rate")
    parser.add_argument("--patience",   type=int,   default=10,         help="Early stopping patience")
    parser.add_argument("--output",     default="efficientnet_fire_best.pth", help="Output checkpoint filename")
    parser.add_argument("--workers",    type=int,   default=0,          help="DataLoader workers")
    args = parser.parse_args()

    set_seed(SEED)
    root = Path(args.dataset).resolve()
    print(f"\n[EfficientNet Trainer] Dataset root : {root}")
    print(f"[EfficientNet Trainer] Epochs        : {args.epochs}")
    print(f"[EfficientNet Trainer] Batch size    : {args.batch_size}")
    print(f"[EfficientNet Trainer] Learning rate : {args.lr}")
    print(f"[EfficientNet Trainer] Output        : {args.output}\n")

    # 1. Validate
    print("[Dataset] Validating...")
    class_report = validate_dataset(root)
    total_valid = sum(v["valid"] for v in class_report.values())
    print(f"[Dataset] Total valid images: {total_valid}")

    # 2. Build CanonicalImageFolder datasets (enforces 0=no_fire, 1=fire)
    full_dataset = CanonicalImageFolder(root=str(root), transform=get_transforms(augment=False))
    folder_to_idx = full_dataset.class_to_idx

    print(f"[Dataset] class_to_idx: {folder_to_idx}")

    # 3. Split (from file if specified, else reproducible stratified split)
    if args.split_file and Path(args.split_file).exists():
        with open(args.split_file, "r") as sf:
            split_data = json.load(sf)
        train_idx = split_data.get("train_indices", [])
        val_idx = split_data.get("val_indices", [])
        test_idx = split_data.get("test_indices", [])
        print(f"\n[Dataset] Loaded pre-defined split from: {args.split_file}")
    else:
        train_idx, val_idx, test_idx = stratified_split(full_dataset)
        print(f"\n[Dataset] Computed fresh stratified split (seed={SEED}):")

    print(f"  Train : {len(train_idx):5d} images")
    print(f"  Val   : {len(val_idx):5d} images")
    print(f"  Test  : {len(test_idx):5d} images")

    train_ds_aug = CanonicalImageFolder(root=str(root), transform=get_transforms(augment=True))

    train_loader = DataLoader(
        Subset(train_ds_aug, train_idx),
        batch_size=args.batch_size,
        sampler=make_sampler(full_dataset, train_idx),
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    # 4. Build model — 2-class head
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Model] Device: {device}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)   # binary: no_fire, fire
    model = model.to(device)

    # Compute inverse class frequency weights for weighted cross-entropy
    train_labels = [full_dataset.targets[i] for i in train_idx]
    c0 = max(1, train_labels.count(0))
    c1 = max(1, train_labels.count(1))
    tot = c0 + c1
    w0 = tot / (2.0 * c0)
    w1 = tot / (2.0 * c1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
    print(f"[Loss] Class weighting applied: no_fire={w0:.3f}, fire={w1:.3f}")

    # Phase 1: freeze backbone, train only classifier
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    freeze_epochs = max(1, args.epochs // 2)

    print(f"\n[Train] Phase 1 (classifier-only, frozen backbone): epochs 1–{freeze_epochs}")
    print(f"[Train] Phase 2 (full fine-tune): epochs {freeze_epochs+1}–{args.epochs}\n")

    for epoch in range(1, args.epochs + 1):
        # Switch to full fine-tuning at halfway point
        if epoch == freeze_epochs + 1:
            print("\n[Train] Unfreezing all layers for full fine-tuning.\n")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - freeze_epochs
            )

        model.train()
        running_loss, running_correct, n_samples = 0.0, 0, 0

        t0 = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            n_samples += inputs.size(0)

        scheduler.step()

        train_loss = running_loss / max(n_samples, 1)
        train_acc = running_correct / max(n_samples, 1) * 100

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss_sum += criterion(outputs, labels).item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += inputs.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1) * 100
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.1f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.1f}% | "
              f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
        })

        # Model selection: maximize val_acc, tie-break with lower val_loss
        is_better = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_to_idx": folder_to_idx,
                "num_classes": 2,
                "classes": CLASSES,
            }, args.output)
            print(f"  → Best model saved (val_acc={val_acc:.1f}%, val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Train] Early stopping triggered: no improvement for {args.patience} epochs.")
                break

    print(f"\n[Train] Done. Best val accuracy: {best_val_acc:.1f}% (val_loss: {best_val_loss:.4f})")
    print(f"[Train] Checkpoint saved to: {args.output}")

    # Save config
    config = {
        "architecture": "efficientnet_b0",
        "num_classes": 2,
        "classes": CLASSES,
        "class_to_idx": folder_to_idx,
        "img_size": IMG_SIZE,
        "seed": SEED,
        "best_val_acc": round(best_val_acc, 2),
        "dataset_root": str(root),
        "split": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
    }
    config_path = Path(args.output).stem + "_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Train] Config saved to: {config_path}")

    # Save training history
    hist_path = "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Train] Training history saved to: {hist_path}")

    print("\n[Train] Test split indices saved in config for use by evaluate_models.py")
    test_split_path = "test_split_indices.json"
    with open(test_split_path, "w") as f:
        json.dump({"test_indices": test_idx, "dataset_root": str(root)}, f)
    print(f"[Train] Test indices saved to: {test_split_path}")


if __name__ == "__main__":
    main()
