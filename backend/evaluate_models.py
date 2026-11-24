"""
FireWatch-AI — Model Evaluation Pipeline
=========================================
Evaluates the actual fire detection models against a labelled test dataset
and computes scientifically valid performance metrics.

Usage
-----
  # Evaluate all models on the held-out test set:
  python evaluate_models.py --dataset ../dataset

  # Evaluate only EfficientNet with a specific checkpoint:
  python evaluate_models.py --dataset ../dataset \\
      --effnet-weights efficientnet_fire_best.pth \\
      --models efficientnet

  # Use pre-saved test-split indices (from train_efficientnet.py):
  python evaluate_models.py --dataset ../dataset \\
      --test-split test_split_indices.json

Outputs (written to evaluation_results/)
-----------------------------------------
  metrics.json              — master metrics for every evaluated model
  classification_report.txt — sklearn classification report (all models)
  confusion_matrix_<model>.png
  roc_curve_<model>.png     — if binary probabilities are available
  per_model_comparison.txt  — markdown table

Important guarantees
---------------------
  - Models are evaluated on the TEST split only. Never on training data.
  - If a model uses random / ImageNet / generic weights, it is marked
    NOT_EVALUATED — its score is NOT included in any summary.
  - Heuristic confidence scores are evaluated separately and clearly
    distinguished from model accuracy.
  - No accuracy numbers are fabricated.
"""

import argparse
import json
import os
import sys
import random
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy imports with availability flags
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    print("[Eval] WARNING: PyTorch not available. Deep-learning models will be skipped.")

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score, roc_curve,
    )
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("[Eval] WARNING: scikit-learn not available. Metrics will be limited.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    print("[Eval] WARNING: matplotlib not available. Plots will be skipped.")

try:
    import seaborn as sns
    _SNS_OK = True
except ImportError:
    _SNS_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 224
CLASSES = ["no_fire", "fire"]   # index 0 = no_fire, 1 = fire
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT_DIR = Path(__file__).parent / "evaluation_results"

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).parent / ".yolo"),
)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_OK:
        torch.manual_seed(seed)


def load_dataset(root: Path):
    """
    Scan dataset root for fire/ and no_fire/ subdirectories.

    Returns
    -------
    image_paths : list of Path
    labels      : list of int  (0 = no_fire, 1 = fire)
    class_counts: dict
    validation_errors: list of str
    """
    image_paths, labels = [], []
    class_counts = {cls: 0 for cls in CLASSES}
    errors = []

    for label_idx, cls_name in enumerate(CLASSES):
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            errors.append(f"Missing class directory: {cls_dir}")
            continue

        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTS:
                continue
            try:
                with Image.open(img_path) as im:
                    im.verify()
                image_paths.append(img_path)
                labels.append(label_idx)
                class_counts[cls_name] += 1
            except Exception as e:
                errors.append(f"Corrupt image (skipped): {img_path} — {e}")

    return image_paths, labels, class_counts, errors


def stratified_split(image_paths, labels, val_frac=0.15, test_frac=0.15, seed=SEED):
    """Fixed-seed stratified split. Returns (train, val, test) index lists."""
    rng = random.Random(seed)
    cls_indices = {}
    for i, lbl in enumerate(labels):
        cls_indices.setdefault(lbl, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for lbl, indices in sorted(cls_indices.items()):
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def get_eval_transform():
    if not _TORCH_OK:
        return None
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_efficientnet(weights_path: str, device):
    """
    Load EfficientNet-B0 for binary fire classification.

    Returns (model, weight_status) where weight_status is one of:
      "custom_fire_weights"  — a checkpoint with 2-class head was found
      "imagenet_fallback"    — no custom weights; ImageNet pretrained used
      "random_weights"       — no weights at all (untrained)
    """
    if not _TORCH_OK:
        return None, "torch_unavailable"

    if weights_path and Path(weights_path).exists():
        try:
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 2)

            ckpt = torch.load(weights_path, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(device)

            if missing:
                print(f"[EfficientNet] Missing keys in checkpoint: {missing[:5]}")
            status = "custom_fire_weights"
            print(f"[EfficientNet] Loaded custom fire weights: {weights_path}")
            return model, status
        except Exception as e:
            print(f"[EfficientNet] Failed to load checkpoint: {e}")

    # Fallback: ImageNet pretrained — NOT suitable for quantitative fire eval
    print("[EfficientNet] No custom weights found. Using ImageNet pretrained (NOT EVALUATED for fire).")
    return None, "imagenet_fallback_no_fire_head"


def load_yolo(model_path: str):
    """
    Load YOLOv8 model and report whether it has fire/smoke classes.

    Returns (model, weight_status, has_fire_classes)
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        class_names = set(str(v).lower() for v in model.names.values())
        has_fire = bool(class_names & {"fire", "smoke", "wildfire"})
        status = "custom_fire_weights" if has_fire else "coco_generic_weights"
        print(f"[YOLOv8] Loaded: {model_path}  |  fire classes present: {has_fire}")
        return model, status, has_fire
    except Exception as e:
        print(f"[YOLOv8] Load failed: {e}")
        return None, "load_failed", False


def load_cnn_lstm(weights_path: str, device):
    """Load CNN-LSTM model. Returns (model, weight_status)."""
    if not _TORCH_OK:
        return None, "torch_unavailable"
    try:
        from models_custom.cnn_lstm import CNNLSTMModel
        model = CNNLSTMModel(num_classes=2, hidden_size=256, num_layers=1)
        if weights_path and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            status = "custom_fire_weights"
            print(f"[CNN-LSTM] Loaded checkpoint: {weights_path}")
        else:
            status = "random_weights"
            print("[CNN-LSTM] No checkpoint found — model has RANDOM WEIGHTS (NOT EVALUATED).")
        model.eval()
        model.to(device)
        return model, status
    except Exception as e:
        print(f"[CNN-LSTM] Load failed: {e}")
        return None, "load_failed"


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_efficientnet(model, image_paths, indices, device, transform, batch_size=32):
    """Return (y_true, y_pred, y_prob_fire) for the given indices."""
    if model is None:
        return None, None, None
    y_true, y_pred, y_prob = [], [], []

    # Build batches
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        tensors, true_labels = [], []
        for i in batch_indices:
            img_path, label = image_paths[i], labels_global[i]
            try:
                img = Image.open(img_path).convert("RGB")
                t = transform(img)
                tensors.append(t)
                true_labels.append(label)
            except Exception as e:
                print(f"[EfficientNet] Skipping {img_path}: {e}")

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        y_true.extend(true_labels)
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs[:, 1].cpu().tolist())   # prob of "fire"

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def predict_yolo_classification(yolo_model, image_paths, indices, labels_all, fire_threshold=0.25):
    """
    Use YOLO as a binary classifier: fire detected if any fire/smoke box found.
    Returns (y_true, y_pred, y_conf).
    NOTE: Only meaningful if model has fire classes (checked before calling).
    """
    y_true, y_pred, y_conf = [], [], []
    for i in indices:
        img_path = image_paths[i]
        true_label = labels_all[i]
        try:
            results = yolo_model.predict(source=str(img_path), conf=fire_threshold, verbose=False)
            fire_found = False
            max_conf = 0.0
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = str(yolo_model.names.get(cls_id, "")).lower()
                    if cls_name in {"fire", "smoke", "wildfire"}:
                        fire_found = True
                        max_conf = max(max_conf, float(box.conf[0].item()))
            pred = 1 if fire_found else 0
            conf = max_conf if fire_found else 0.0
        except Exception as e:
            pred = 0
            conf = 0.0

        y_true.append(true_label)
        y_pred.append(pred)
        y_conf.append(conf)

    return np.array(y_true), np.array(y_pred), np.array(y_conf)


def predict_heuristic(image_paths, indices, labels_all):
    """
    Evaluate the color-histogram heuristic from image_analyzer.py as a classifier.
    Classified as 'fire' if heuristic fire_color score >= threshold.
    Returns (y_true, y_pred, y_conf).
    """
    # Import inline to avoid circular deps
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from image_analyzer import ImageAnalyzer
        analyzer = ImageAnalyzer()
    except Exception as e:
        print(f"[Heuristic] Cannot load ImageAnalyzer: {e}")
        return None, None, None

    y_true, y_pred, y_conf = [], [], []
    for i in indices:
        img_path = image_paths[i]
        true_label = labels_all[i]
        try:
            with open(img_path, "rb") as f:
                result = analyzer.analyze_image(f.read())
            fire_detected = result.get("fire_detected", False)
            conf = result.get("confidence", 0.0)
            pred = 1 if fire_detected else 0
        except Exception as e:
            pred = 0
            conf = 0.0

        y_true.append(true_label)
        y_pred.append(pred)
        y_conf.append(conf)

    return np.array(y_true), np.array(y_pred), np.array(y_conf)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob=None, model_name="model"):
    """Compute and return a metrics dict. Never fabricates values."""
    if not _SKLEARN_OK:
        return {"error": "scikit-learn unavailable"}
    if len(y_true) == 0:
        return {"error": "empty test set"}

    n = len(y_true)
    n_fire = int(np.sum(y_true == 1))
    n_no_fire = int(np.sum(y_true == 0))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES, zero_division=0,
        output_dict=False,
    )

    roc_auc = None
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass

    result = {
        "model": model_name,
        "n_test_samples": n,
        "n_fire_samples": n_fire,
        "n_no_fire_samples": n_no_fire,
        "accuracy": round(float(acc), 4),
        "precision_fire": round(float(prec), 4),
        "recall_fire": round(float(rec), 4),
        "f1_fire": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else "N/A",
        "confusion_matrix": {
            "labels": CLASSES,
            "matrix": cm,
            "note": "rows=actual, cols=predicted",
        },
        "classification_report_text": report,
    }
    return result


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm_data, model_name: str):
    if not _MPL_OK:
        return
    OUT_DIR.mkdir(exist_ok=True)

    cm = np.array(cm_data["matrix"])
    labels = cm_data["labels"]

    fig, ax = plt.subplots(figsize=(5, 4))
    if _SNS_OK:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                    xticklabels=labels, yticklabels=labels, ax=ax)
    else:
        im = ax.imshow(cm, cmap="Reds")
        plt.colorbar(im, ax=ax)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    fname = OUT_DIR / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"[Plot] Saved: {fname}")


def plot_roc_curve(y_true, y_prob, model_name: str):
    if not (_MPL_OK and _SKLEARN_OK):
        return
    if y_prob is None or len(np.unique(y_true)) < 2:
        return
    OUT_DIR.mkdir(exist_ok=True)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {model_name}")
        ax.legend()
        plt.tight_layout()
        fname = OUT_DIR / f"roc_curve_{model_name.replace(' ', '_')}.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"[Plot] Saved: {fname}")
    except Exception as e:
        print(f"[Plot] ROC curve failed for {model_name}: {e}")


# ---------------------------------------------------------------------------
# Global labels reference (populated in main)
# ---------------------------------------------------------------------------
labels_global = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global labels_global

    parser = argparse.ArgumentParser(description="FireWatch-AI Model Evaluation Pipeline")
    parser.add_argument("--dataset",       default="../dataset",           help="Root dataset directory (fire/ and no_fire/)")
    parser.add_argument("--effnet-weights",default="efficientnet_fire_best.pth", help="EfficientNet checkpoint path")
    parser.add_argument("--yolo-weights",  default="yolov8n.pt",           help="YOLO weights path")
    parser.add_argument("--lstm-weights",  default="cnn_lstm_best.pth",    help="CNN-LSTM checkpoint path")
    parser.add_argument("--test-split",    default=None,                   help="JSON file with test_indices (from train_efficientnet.py)")
    parser.add_argument("--models",        default="all",
                        help="Comma-separated subset to evaluate: efficientnet,yolo,cnn_lstm,heuristic,all")
    parser.add_argument("--batch-size",    type=int, default=16,           help="Batch size for inference")
    args = parser.parse_args()

    set_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    enabled_models = set(args.models.lower().split(","))
    if "all" in enabled_models:
        enabled_models = {"efficientnet", "yolo", "cnn_lstm", "heuristic"}

    root = Path(args.dataset).resolve()
    print(f"\n{'='*60}")
    print("FireWatch-AI — Model Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"Dataset root   : {root}")
    print(f"EfficientNet   : {args.effnet_weights}")
    print(f"YOLO           : {args.yolo_weights}")
    print(f"CNN-LSTM       : {args.lstm_weights}")
    print(f"Models         : {enabled_models}")
    print(f"Output dir     : {OUT_DIR}")
    print(f"{'='*60}\n")

    # 1. Load and validate dataset
    print("[Dataset] Scanning and validating...")
    image_paths, labels, class_counts, errors = load_dataset(root)

    if errors:
        print(f"\n[Dataset] Issues found ({len(errors)}):")
        for e in errors:
            print(f"  ⚠  {e}")

    if not image_paths:
        print("\n[Dataset] ERROR: No valid images found.")
        print("Please populate the dataset directory:")
        print(f"  {root}/fire/      ← fire images")
        print(f"  {root}/no_fire/   ← non-fire images")
        print("\nSuggested public dataset sources:")
        print("  • AIDER dataset: https://zenodo.org/records/3888147")
        print("  • Fire vs No Fire (Kaggle): https://www.kaggle.com/datasets/phylake1337/fire-dataset")
        print("  • FiSmo dataset: https://github.com/mtcazzolato/smokeandfire")
        _write_empty_report(root, class_counts, errors)
        sys.exit(0)

    labels_global = labels

    # 2. Split
    if args.test_split and Path(args.test_split).exists():
        with open(args.test_split) as f:
            split_data = json.load(f)
        test_idx = split_data.get("test_indices", [])
        # Validate indices are in range
        test_idx = [i for i in test_idx if i < len(image_paths)]
        print(f"[Dataset] Using pre-saved test split: {len(test_idx)} samples")
    else:
        train_idx, val_idx, test_idx = stratified_split(image_paths, labels)
        print(f"[Dataset] Fresh stratified split (seed={SEED}):")
        print(f"  Train : {len(train_idx):5d}")
        print(f"  Val   : {len(val_idx):5d}")
        print(f"  Test  : {len(test_idx):5d}")

    total = class_counts.get("fire", 0) + class_counts.get("no_fire", 0)
    print(f"\n[Dataset] Class distribution (all images):")
    for cls in CLASSES:
        cnt = class_counts.get(cls, 0)
        pct = cnt / max(total, 1) * 100
        print(f"  {cls:10s}: {cnt:5d} ({pct:.1f}%)")

    test_labels = [labels[i] for i in test_idx]
    n_test_fire = sum(1 for l in test_labels if l == 1)
    n_test_nofire = sum(1 for l in test_labels if l == 0)
    print(f"\n[Dataset] Test split ({len(test_idx)} samples): "
          f"{n_test_fire} fire, {n_test_nofire} no_fire")

    if len(test_idx) == 0:
        print("[Eval] ERROR: Test split is empty. Need more images.")
        sys.exit(1)

    device = torch.device("cpu") if _TORCH_OK else None
    transform = get_eval_transform()

    all_metrics = {}
    not_evaluated = []
    audit = {}

    # ------------------------------------------------------------------
    # Evaluate EfficientNet
    # ------------------------------------------------------------------
    if "efficientnet" in enabled_models:
        print(f"\n{'─'*50}")
        print("[EfficientNet] Loading...")
        eff_model, eff_status = load_efficientnet(args.effnet_weights, device)

        audit["efficientnet"] = {
            "architecture": "EfficientNet-B0 (binary fire/no_fire head)",
            "weights_path": args.effnet_weights,
            "weight_status": eff_status,
            "can_detect_fire": eff_status == "custom_fire_weights",
            "evaluation_status": "EVALUATED" if eff_status == "custom_fire_weights" else "NOT_EVALUATED",
        }

        if eff_status == "custom_fire_weights" and eff_model is not None:
            print("[EfficientNet] Running inference on test set...")
            y_true, y_pred, y_prob = predict_efficientnet(
                eff_model, image_paths, test_idx, device, transform, args.batch_size
            )
            if y_true is not None and len(y_true) > 0:
                m = compute_metrics(y_true, y_pred, y_prob, "EfficientNet-B0")
                all_metrics["efficientnet"] = m
                plot_confusion_matrix(m["confusion_matrix"], "EfficientNet-B0")
                plot_roc_curve(y_true, y_prob, "EfficientNet-B0")
                print(f"[EfficientNet] Accuracy : {m['accuracy']*100:.1f}%")
                print(f"[EfficientNet] F1 (fire): {m['f1_fire']*100:.1f}%")
            else:
                not_evaluated.append("EfficientNet-B0: inference returned no results")
        else:
            reason = (
                "Using ImageNet pretrained weights with 1000-class head — "
                "no fire/no_fire binary classifier. NOT EVALUATED."
                if "imagenet" in eff_status
                else eff_status
            )
            not_evaluated.append(f"EfficientNet-B0: {reason}")
            print(f"[EfficientNet] NOT EVALUATED — {reason}")

    # ------------------------------------------------------------------
    # Evaluate YOLOv8
    # ------------------------------------------------------------------
    if "yolo" in enabled_models:
        print(f"\n{'─'*50}")
        print("[YOLOv8] Loading...")
        yolo_model, yolo_status, has_fire = load_yolo(args.yolo_weights)

        audit["yolov8"] = {
            "architecture": "YOLOv8n",
            "weights_path": args.yolo_weights,
            "weight_status": yolo_status,
            "has_fire_classes": has_fire,
            "can_detect_fire": has_fire,
            "evaluation_status": "EVALUATED" if has_fire else "NOT_EVALUATED",
        }

        if has_fire and yolo_model is not None:
            print("[YOLOv8] Fire classes found. Running inference on test set...")
            y_true, y_pred, y_conf = predict_yolo_classification(
                yolo_model, image_paths, test_idx, labels
            )
            if y_true is not None and len(y_true) > 0:
                m = compute_metrics(y_true, y_pred, y_conf, "YOLOv8")
                all_metrics["yolov8"] = m
                plot_confusion_matrix(m["confusion_matrix"], "YOLOv8")
                print(f"[YOLOv8] Accuracy : {m['accuracy']*100:.1f}%")
                print(f"[YOLOv8] F1 (fire): {m['f1_fire']*100:.1f}%")
        else:
            reason = (
                "yolov8n.pt uses 80 generic COCO classes — 'fire' and 'smoke' are NOT present. "
                "It cannot detect fire. NOT EVALUATED."
                if yolo_model is not None
                else "YOLO model failed to load."
            )
            not_evaluated.append(f"YOLOv8: {reason}")
            print(f"[YOLOv8] NOT EVALUATED — {reason}")

    # ------------------------------------------------------------------
    # Evaluate CNN-LSTM
    # ------------------------------------------------------------------
    if "cnn_lstm" in enabled_models:
        print(f"\n{'─'*50}")
        print("[CNN-LSTM] Loading...")
        lstm_model, lstm_status = load_cnn_lstm(args.lstm_weights, device)

        audit["cnn_lstm"] = {
            "architecture": "EfficientNet-B0 CNN + LSTM (seq_length=5)",
            "weights_path": args.lstm_weights,
            "weight_status": lstm_status,
            "can_detect_fire": lstm_status == "custom_fire_weights",
            "evaluation_status": "EVALUATED" if lstm_status == "custom_fire_weights" else "NOT_EVALUATED",
            "note": (
                "CNN-LSTM requires temporal sequences of frames from the same scene. "
                "Still-image datasets cannot substitute for genuine temporal sequences."
            ),
        }

        if lstm_status == "custom_fire_weights" and lstm_model is not None:
            # Evaluate: use single-frame sequences (seq_length=1) on still images
            # This is methodologically imperfect but transparent
            print("[CNN-LSTM] Evaluating with single-image sequences "
                  "(still images, not true temporal sequences — results will be conservative).")
            y_true_lst, y_pred_lst, y_prob_lst = [], [], []
            for i in test_idx:
                img_path = image_paths[i]
                true_label = labels[i]
                try:
                    img = Image.open(img_path).convert("RGB")
                    t = transform(img)
                    # Duplicate frame to satisfy minimum seq_length=5
                    seq = torch.stack([t] * 5).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = lstm_model(seq)
                        probs = torch.softmax(logits, dim=1)[0]
                        fire_prob = float(probs[1].item())
                    pred = 1 if fire_prob >= 0.5 else 0
                    y_true_lst.append(true_label)
                    y_pred_lst.append(pred)
                    y_prob_lst.append(fire_prob)
                except Exception as e:
                    pass

            if y_true_lst:
                m = compute_metrics(
                    np.array(y_true_lst), np.array(y_pred_lst),
                    np.array(y_prob_lst), "CNN-LSTM"
                )
                m["note"] = ("Evaluated on still images duplicated to seq_length=5. "
                             "Not equivalent to true temporal evaluation.")
                all_metrics["cnn_lstm"] = m
                plot_confusion_matrix(m["confusion_matrix"], "CNN-LSTM")
                plot_roc_curve(np.array(y_true_lst), np.array(y_prob_lst), "CNN-LSTM")
                print(f"[CNN-LSTM] Accuracy : {m['accuracy']*100:.1f}%")
                print(f"[CNN-LSTM] F1 (fire): {m['f1_fire']*100:.1f}%")
            else:
                not_evaluated.append("CNN-LSTM: inference produced no results")
        else:
            not_evaluated.append(
                f"CNN-LSTM: {lstm_status}. "
                "Must be trained with train_cnn_lstm.py using genuine temporal sequence data."
            )
            print(f"[CNN-LSTM] NOT EVALUATED — {lstm_status}")

    # ------------------------------------------------------------------
    # Evaluate Heuristic
    # ------------------------------------------------------------------
    if "heuristic" in enabled_models:
        print(f"\n{'─'*50}")
        print("[Heuristic] Running color-histogram heuristic on test set...")
        y_true_h, y_pred_h, y_conf_h = predict_heuristic(image_paths, test_idx, labels)

        audit["heuristic"] = {
            "architecture": "Color-histogram + brightness + texture heuristic",
            "weights_path": "N/A",
            "weight_status": "rule_based",
            "can_detect_fire": True,
            "evaluation_status": "EVALUATED" if y_true_h is not None else "FAILED",
            "note": "Heuristic scores are rule-based thresholds, NOT model accuracy.",
        }

        if y_true_h is not None and len(y_true_h) > 0:
            m = compute_metrics(y_true_h, y_pred_h, None, "Heuristic")
            m["note"] = ("Heuristic rule-based classifier. "
                         "Scores are computed from color statistics, not learned from data. "
                         "Must be compared against model-based classifiers separately.")
            all_metrics["heuristic"] = m
            plot_confusion_matrix(m["confusion_matrix"], "Heuristic")
            print(f"[Heuristic] Accuracy : {m['accuracy']*100:.1f}%")
            print(f"[Heuristic] F1 (fire): {m['f1_fire']*100:.1f}%")
        else:
            not_evaluated.append("Heuristic: failed to run (check image_analyzer.py imports)")

    # ------------------------------------------------------------------
    # Save master metrics.json
    # ------------------------------------------------------------------
    master = {
        "evaluation_date": datetime.now(UTC).isoformat(),
        "dataset": {
            "root": str(root),
            "total_images": len(image_paths),
            "class_distribution": class_counts,
            "test_set_size": len(test_idx),
            "split_seed": SEED,
            "validation_errors": errors,
        },
        "model_audit": audit,
        "metrics": all_metrics,
        "not_evaluated": not_evaluated,
    }

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(master, f, indent=2)
    print(f"\n[Eval] Metrics saved to: {metrics_path}")

    # ------------------------------------------------------------------
    # Classification report text
    # ------------------------------------------------------------------
    report_lines = []
    report_lines.append("FireWatch-AI — Evaluation Classification Report")
    report_lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    report_lines.append(f"Dataset  : {root}")
    report_lines.append(f"Test set : {len(test_idx)} samples  "
                        f"(fire={n_test_fire}, no_fire={n_test_nofire})")
    report_lines.append("")

    for model_key, m in all_metrics.items():
        report_lines.append(f"{'='*55}")
        report_lines.append(f"Model: {m.get('model', model_key)}")
        if m.get("note"):
            report_lines.append(f"NOTE : {m['note']}")
        report_lines.append(f"{'='*55}")
        report_lines.append(m.get("classification_report_text", ""))
        report_lines.append(f"  Accuracy  : {m.get('accuracy', 'N/A')}")
        report_lines.append(f"  Precision : {m.get('precision_fire', 'N/A')}")
        report_lines.append(f"  Recall    : {m.get('recall_fire', 'N/A')}")
        report_lines.append(f"  F1-score  : {m.get('f1_fire', 'N/A')}")
        report_lines.append(f"  ROC-AUC   : {m.get('roc_auc', 'N/A')}")
        report_lines.append("")

    if not_evaluated:
        report_lines.append(f"{'='*55}")
        report_lines.append("MODELS NOT EVALUATED (reason):")
        for reason in not_evaluated:
            report_lines.append(f"  ✗  {reason}")

    report_text = "\n".join(report_lines)
    report_path = OUT_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"[Eval] Classification report saved to: {report_path}")

    # ------------------------------------------------------------------
    # Per-model comparison table
    # ------------------------------------------------------------------
    comparison_lines = ["# FireWatch-AI — Per-Model Comparison\n"]
    comparison_lines.append(f"**Test set:** {len(test_idx)} images "
                             f"({n_test_fire} fire, {n_test_nofire} no_fire)\n")
    comparison_lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Status |")
    comparison_lines.append("|-------|----------|-----------|--------|----|---------|--------|")

    for model_key, m in all_metrics.items():
        acc = f"{m.get('accuracy', 0)*100:.1f}%"
        prec = f"{m.get('precision_fire', 0)*100:.1f}%"
        rec = f"{m.get('recall_fire', 0)*100:.1f}%"
        f1 = f"{m.get('f1_fire', 0)*100:.1f}%"
        auc = str(m.get("roc_auc", "N/A"))
        status = audit.get(model_key, {}).get("evaluation_status", "EVALUATED")
        comparison_lines.append(
            f"| {m.get('model', model_key)} | {acc} | {prec} | {rec} | {f1} | {auc} | {status} |"
        )

    for reason in not_evaluated:
        name = reason.split(":")[0].strip()
        comparison_lines.append(f"| {name} | — | — | — | — | — | NOT EVALUATED |")

    comparison_lines.append("")
    comparison_lines.append("## Model Audit Summary\n")
    for mkey, info in audit.items():
        comparison_lines.append(f"### {mkey}")
        for k, v in info.items():
            comparison_lines.append(f"- **{k}**: {v}")
        comparison_lines.append("")

    comparison_text = "\n".join(comparison_lines)
    comp_path = OUT_DIR / "per_model_comparison.md"
    with open(comp_path, "w") as f:
        f.write(comparison_text)
    print(f"[Eval] Comparison table saved to: {comp_path}")

    # ------------------------------------------------------------------
    # Summary printout
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    if all_metrics:
        for mkey, m in all_metrics.items():
            print(f"  {m.get('model', mkey):25s}  "
                  f"Acc={m.get('accuracy',0)*100:.1f}%  "
                  f"F1={m.get('f1_fire',0)*100:.1f}%  "
                  f"AUC={m.get('roc_auc','N/A')}")
    else:
        print("  No models evaluated with real fire weights.")

    if not_evaluated:
        print(f"\nNOT EVALUATED ({len(not_evaluated)}):")
        for r in not_evaluated:
            print(f"  ✗  {r[:90]}")

    print(f"\n[Eval] All outputs in: {OUT_DIR}/")
    print(f"{'='*60}\n")


def _write_empty_report(root, class_counts, errors):
    """Write a placeholder report when no images are available."""
    OUT_DIR.mkdir(exist_ok=True)
    report = {
        "evaluation_date": datetime.now(UTC).isoformat(),
        "status": "NO_DATASET",
        "message": (
            "No labelled images found. Please populate the dataset directory "
            "and re-run evaluation."
        ),
        "dataset_root": str(root),
        "class_distribution": class_counts,
        "validation_errors": errors,
        "suggested_datasets": [
            "AIDER: https://zenodo.org/records/3888147",
            "Fire vs No Fire (Kaggle): https://www.kaggle.com/datasets/phylake1337/fire-dataset",
            "FiSmo: https://github.com/mtcazzolato/smokeandfire",
        ],
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Eval] Placeholder report written to: {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
