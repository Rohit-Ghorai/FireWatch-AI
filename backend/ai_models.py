"""
Hybrid AI model engine for fire detection.
Supports YOLOv8 object detection, EfficientNet classification, and CNN-LSTM temporal sequences.
Provides transparent, auditable model status distinguishing custom-trained weights from fallbacks.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
    from torchvision import models, transforms
except Exception:
    torch = None
    models = None
    transforms = None

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yolo"))

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class AIModelEngine:
    """Runs YOLOv8, EfficientNet, and CNN-LSTM inference with transparent status reporting."""

    def __init__(self):
        self.device = "cpu"
        if torch is not None and torch.cuda.is_available():
            self.device = "cuda"

        self.yolo_model = None
        self.effnet_model = None
        self.effnet_transform = None
        self.scene_labels = [
            label.strip().lower()
            for label in os.getenv("SCENE_CLASSES", "fire,sunlight,smoke,fog,normal").split(",")
            if label.strip()
        ]

        self.yolo_labels = {
            label.strip().lower()
            for label in os.getenv("YOLO_FIRE_CLASS_NAMES", "fire,smoke,wildfire").split(",")
            if label.strip()
        }
        self.yolo_conf_threshold = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))

        self.effnet_enabled = os.getenv("ENABLE_EFFICIENTNET", "true").lower() == "true"
        self.yolo_enabled = os.getenv("ENABLE_YOLOV8", "true").lower() == "true"

        self.model_status: Dict[str, Dict[str, Any]] = {
            "yolov8": {},
            "efficientnet": {},
            "cnn_lstm": {},
        }

        self.frame_buffer = []
        self.seq_length = 5
        self.cnn_lstm_model = None

        self._load_yolo()
        self._load_efficientnet()
        self._load_cnn_lstm()

    def _get_evaluation_metrics(self, model_key: str) -> Dict[str, Any]:
        """Read evaluation metrics from evaluation_results/metrics.json if present."""
        metrics_file = Path(__file__).parent / "evaluation_results" / "metrics.json"
        if not metrics_file.exists():
            return None
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
            return data.get("metrics", {}).get(model_key)
        except Exception:
            return None

    def _load_cnn_lstm(self):
        eval_metrics = self._get_evaluation_metrics("cnn_lstm")
        status = {
            "enabled": True,
            "available": False,
            "loaded": False,
            "custom_weights_loaded": False,
            "pretrained_fallback": False,
            "random_weights": False,
            "can_detect_fire": False,
            "evaluation_status": "evaluated" if eval_metrics else "not_evaluated",
            "metrics": eval_metrics,
            "ready": False,
            "error": None,
            "note": "",
        }

        try:
            from models_custom.cnn_lstm import CNNLSTMModel
            if torch is None:
                status["error"] = "torch_not_installed"
                status["note"] = "PyTorch is required for CNN-LSTM temporal model."
                self.model_status["cnn_lstm"] = status
                return

            status["available"] = True
            self.cnn_lstm_model = CNNLSTMModel(num_classes=2, hidden_size=256, num_layers=1)

            raw_path = (os.getenv("CNN_LSTM_WEIGHTS_PATH") or "").strip() or "cnn_lstm_best.pth"
            weights_path = os.path.join(os.path.dirname(__file__), raw_path) if not os.path.isabs(raw_path) else raw_path
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                self.cnn_lstm_model.load_state_dict(state_dict, strict=False)
                self.cnn_lstm_model.eval()
                self.cnn_lstm_model.to(self.device)
                status["loaded"] = True
                status["custom_weights_loaded"] = True
                status["can_detect_fire"] = True
                status["ready"] = True
                status["weights_path"] = weights_path
                status["note"] = f"Custom-trained CNN-LSTM model loaded from {weights_path}."
            else:
                self.cnn_lstm_model.eval()
                self.cnn_lstm_model.to(self.device)
                status["loaded"] = True
                status["random_weights"] = True
                status["ready"] = False
                status["error"] = "using_random_weights"
                status["note"] = "Model architecture initialized with random weights. Train using train_cnn_lstm.py."
        except Exception as e:
            status["error"] = str(e)
            status["note"] = f"Failed to initialize CNN-LSTM: {e}"

        self.model_status["cnn_lstm"] = status

    def _load_yolo(self):
        eval_metrics = self._get_evaluation_metrics("yolov8")
        status = {
            "enabled": self.yolo_enabled,
            "available": YOLO is not None,
            "loaded": False,
            "custom_weights_loaded": False,
            "pretrained_fallback": False,
            "random_weights": False,
            "can_detect_fire": False,
            "evaluation_status": "evaluated" if eval_metrics else "not_evaluated",
            "metrics": eval_metrics,
            "ready": False,
            "error": None,
            "note": "",
        }

        if not self.yolo_enabled:
            status["note"] = "YOLOv8 is disabled in configuration."
            self.model_status["yolov8"] = status
            return

        if YOLO is None:
            status["error"] = "ultralytics_not_installed"
            status["note"] = "ultralytics package not installed."
            self.model_status["yolov8"] = status
            return

        try:
            raw_path = (os.getenv("YOLOV8_MODEL_PATH") or "").strip() or "yolov8n.pt"
            model_path = os.path.join(os.path.dirname(__file__), raw_path) if not os.path.isabs(raw_path) else raw_path
            self.yolo_model = YOLO(model_path)
            status["loaded"] = True
            status["ready"] = True
            status["weights_path"] = model_path

            class_names = set(str(v).lower() for v in self.yolo_model.names.values())
            has_fire_classes = bool(class_names & self.yolo_labels)

            if has_fire_classes:
                status["custom_weights_loaded"] = True
                status["can_detect_fire"] = True
                status["note"] = f"Custom YOLO model with fire/smoke classes loaded ({model_path})."
            else:
                status["pretrained_fallback"] = True
                status["can_detect_fire"] = False
                status["error"] = "coco_generic_classes_no_fire"
                status["note"] = (
                    f"Pretrained COCO model in use ({model_path}, 80 generic classes). "
                    "Classes 'fire'/'smoke' are not present; model cannot genuinely detect fire."
                )
        except Exception as e:
            status["error"] = str(e)
            status["note"] = f"Failed to load YOLO model: {e}"

        self.model_status["yolov8"] = status

    def _load_efficientnet(self):
        eval_metrics = self._get_evaluation_metrics("efficientnet")
        status = {
            "enabled": self.effnet_enabled,
            "available": torch is not None and models is not None,
            "loaded": False,
            "custom_weights_loaded": False,
            "pretrained_fallback": False,
            "random_weights": False,
            "can_detect_fire": False,
            "evaluation_status": "evaluated" if eval_metrics else "not_evaluated",
            "metrics": eval_metrics,
            "ready": False,
            "error": None,
            "note": "",
        }

        if not self.effnet_enabled:
            status["note"] = "EfficientNet is disabled in configuration."
            self.model_status["efficientnet"] = status
            return

        if torch is None or models is None or transforms is None:
            status["error"] = "torch_or_torchvision_not_installed"
            status["note"] = "PyTorch or torchvision is not installed."
            self.model_status["efficientnet"] = status
            return

        self.effnet_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        try:
            raw_path = (os.getenv("EFFICIENTNET_WEIGHTS_PATH") or "").strip() or "efficientnet_fire_best.pth"
            weights_path = os.path.join(os.path.dirname(__file__), raw_path) if not os.path.isabs(raw_path) else raw_path
            if os.path.exists(weights_path):
                self.effnet_model = models.efficientnet_b0(weights=None)
                in_features = self.effnet_model.classifier[1].in_features
                self.effnet_model.classifier[1] = torch.nn.Linear(in_features, 2)

                checkpoint = torch.load(weights_path, map_location=self.device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                self.effnet_model.load_state_dict(state_dict, strict=False)
                self.effnet_model.eval()
                self.effnet_model.to(self.device)

                status["loaded"] = True
                status["custom_weights_loaded"] = True
                status["can_detect_fire"] = True
                status["ready"] = True
                status["weights_path"] = weights_path
                status["note"] = f"Custom fire/no-fire fine-tuned weights loaded from {weights_path}."
            else:
                self.effnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                self.effnet_model.eval()
                self.effnet_model.to(self.device)

                status["loaded"] = True
                status["pretrained_fallback"] = True
                status["can_detect_fire"] = False
                status["ready"] = True
                status["error"] = "using_imagenet_fallback"
                status["note"] = (
                    "ImageNet-1K pretrained weights in use (1000 generic classes). "
                    "No fire classification head; fine-tune with train_efficientnet.py."
                )
        except Exception as e:
            status["error"] = str(e)
            status["note"] = f"Failed to load EfficientNet: {e}"

        self.model_status["efficientnet"] = status

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive status for all models with audit flags and evaluation summary."""
        for key in ["efficientnet", "yolov8", "cnn_lstm"]:
            eval_metrics = self._get_evaluation_metrics(key)
            if eval_metrics:
                self.model_status[key]["evaluation_status"] = "evaluated"
                self.model_status[key]["metrics"] = {
                    "accuracy": eval_metrics.get("accuracy"),
                    "precision": eval_metrics.get("precision_fire"),
                    "recall": eval_metrics.get("recall_fire"),
                    "f1_score": eval_metrics.get("f1_fire"),
                    "roc_auc": eval_metrics.get("roc_auc"),
                    "test_samples": eval_metrics.get("n_test_samples"),
                }

        heuristic_metrics = self._get_evaluation_metrics("heuristic")

        return {
            "device": self.device,
            "models": self.model_status,
            "heuristic_evaluation": {
                "evaluation_status": "evaluated" if heuristic_metrics else "not_evaluated",
                "metrics": {
                    "accuracy": heuristic_metrics.get("accuracy"),
                    "precision": heuristic_metrics.get("precision_fire"),
                    "recall": heuristic_metrics.get("recall_fire"),
                    "f1_score": heuristic_metrics.get("f1_fire"),
                    "test_samples": heuristic_metrics.get("n_test_samples"),
                } if heuristic_metrics else None,
                "note": "Rule-based color histogram and texture heuristic. Clearly separated from learned models.",
            },
        }

    def predict(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Runs model inference only on models that possess genuine custom weights.
        Untrained/random/generic models are excluded from the ensemble to prevent false confidence.
        """
        import cv2
        resized_frame = cv2.resize(img_array, (224, 224))
        self.frame_buffer.append(resized_frame)
        if len(self.frame_buffer) > self.seq_length:
            self.frame_buffer.pop(0)

        yolo_result = self._predict_yolo(img_array)
        effnet_result = self._predict_efficientnet(img_array)
        lstm_result = self._predict_cnn_lstm()

        trained_scores = []
        if yolo_result.get("available") and self.model_status["yolov8"].get("can_detect_fire"):
            trained_scores.append(("yolov8", yolo_result.get("confidence", 0.0), 0.50))
        if effnet_result.get("available") and self.model_status["efficientnet"].get("can_detect_fire"):
            trained_scores.append(("efficientnet", effnet_result.get("confidence", 0.0), 0.50))
        if (lstm_result.get("available") and self.model_status["cnn_lstm"].get("can_detect_fire")
                and len(self.frame_buffer) == self.seq_length):
            trained_scores.append(("cnn_lstm", lstm_result.get("confidence", 0.0), 0.30))

        if not trained_scores:
            return {
                "model_used": False,
                "confidence": 0.0,
                "fire_detected": False,
                "scene_classification": "unknown",
                "regions": [],
                "note": "No custom-trained fire detection weights loaded. Relying on heuristic analysis.",
                "details": {
                    "yolov8": yolo_result,
                    "efficientnet": effnet_result,
                    "cnn_lstm": lstm_result,
                },
            }

        total_weight = sum(w for _, _, w in trained_scores)
        ensemble_conf = sum(score * (w / total_weight) for _, score, w in trained_scores)
        ensemble_conf = round(float(max(0.0, min(99.5, ensemble_conf))), 1)

        scene_class = effnet_result.get("predicted_class", "fire" if ensemble_conf >= 50 else "normal")
        fire_detected = ensemble_conf >= 50 or (effnet_result.get("fire_detected", False))

        return {
            "model_used": True,
            "confidence": ensemble_conf,
            "fire_detected": fire_detected,
            "scene_classification": scene_class,
            "regions": yolo_result.get("regions", []),
            "details": {
                "yolov8": yolo_result,
                "efficientnet": effnet_result,
                "cnn_lstm": lstm_result,
            },
        }

    def _predict_cnn_lstm(self) -> Dict[str, Any]:
        if (self.cnn_lstm_model is None or len(self.frame_buffer) < self.seq_length
                or torch is None or not self.model_status["cnn_lstm"].get("can_detect_fire")):
            reason = "random_weights_not_evaluated" if self.model_status["cnn_lstm"].get("random_weights") else "not_loaded_or_buffering"
            return {"available": False, "confidence": 0.0, "reason": reason}

        try:
            from PIL import Image

            tensors = []
            for frame in self.frame_buffer:
                img = Image.fromarray(frame).convert("RGB")
                if self.effnet_transform:
                    t = self.effnet_transform(img)
                else:
                    t = torch.zeros(3, 224, 224)
                tensors.append(t)

            input_tensor = torch.stack(tensors).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.cnn_lstm_model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                fire_prob = float(probs[1].item())
                confidence = round(fire_prob * 100, 1)

            return {
                "available": True,
                "confidence": confidence,
                "predicted_class": "fire" if fire_prob >= 0.5 else "normal",
            }
        except Exception as e:
            return {"available": False, "confidence": 0.0, "reason": str(e)}

    def _predict_yolo(self, img_array: np.ndarray) -> Dict[str, Any]:
        if self.yolo_model is None:
            return {"available": False, "confidence": 0.0, "regions": [], "reason": "not_loaded"}

        if not self.model_status["yolov8"].get("can_detect_fire"):
            return {
                "available": True,
                "confidence": 0.0,
                "regions": [],
                "detections": 0,
                "reason": "coco_generic_classes_no_fire",
                "can_detect_fire": False,
            }

        try:
            results = self.yolo_model.predict(
                source=img_array,
                conf=self.yolo_conf_threshold,
                verbose=False,
                device=self.device,
            )

            fire_regions: List[Dict[str, float]] = []
            confidences: List[float] = []
            h, w = img_array.shape[:2]

            for result in results:
                names = result.names
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                    label = str(names.get(cls_id, "unknown")).lower()
                    conf = float(box.conf[0].item()) if box.conf is not None else 0.0

                    if label not in self.yolo_labels:
                        continue

                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    region = {
                        "x": round(max(0.0, (x1 / max(w, 1)) * 100), 2),
                        "y": round(max(0.0, (y1 / max(h, 1)) * 100), 2),
                        "width": round(max(0.0, ((x2 - x1) / max(w, 1)) * 100), 2),
                        "height": round(max(0.0, ((y2 - y1) / max(h, 1)) * 100), 2),
                        "density": round(conf * 100, 1),
                        "label": label,
                    }
                    fire_regions.append(region)
                    confidences.append(conf)

            confidence = round((max(confidences) * 100), 1) if confidences else 0.0
            return {
                "available": True,
                "confidence": confidence,
                "regions": fire_regions,
                "detections": len(fire_regions),
                "can_detect_fire": True,
            }
        except Exception as e:
            return {"available": False, "confidence": 0.0, "regions": [], "reason": str(e)}

    def _predict_efficientnet(self, img_array: np.ndarray) -> Dict[str, Any]:
        if self.effnet_model is None or self.effnet_transform is None or torch is None:
            return {"available": False, "confidence": 0.0, "reason": "not_loaded"}

        if not self.model_status["efficientnet"].get("can_detect_fire"):
            return {
                "available": True,
                "confidence": 0.0,
                "mode": "imagenet_fallback",
                "predicted_class": "normal",
                "fire_detected": False,
                "reason": "using_imagenet_fallback_cannot_detect_fire",
                "can_detect_fire": False,
            }

        try:
            input_tensor = self.effnet_transform(img_array).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.effnet_model(input_tensor)

            probs = torch.softmax(logits, dim=1)[0]
            fire_prob = float(probs[1].item())
            confidence = round(fire_prob * 100, 1)

            return {
                "available": True,
                "confidence": confidence,
                "mode": "binary_fire_classifier",
                "predicted_class": "fire" if fire_prob >= 0.5 else "normal",
                "fire_detected": fire_prob >= 0.5,
                "class_scores": {
                    "no_fire": round((1 - fire_prob) * 100, 1),
                    "fire": confidence,
                },
                "can_detect_fire": True,
            }
        except Exception as e:
            return {"available": False, "confidence": 0.0, "reason": str(e)}
