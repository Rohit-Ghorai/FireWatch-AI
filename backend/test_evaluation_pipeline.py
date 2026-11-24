"""
Unit tests for the AI Model Evaluation Pipeline and Status Reporting.
Tests:
  - Metric computation mathematical correctness (Accuracy, Precision, Recall, F1)
  - Dataset validation and split stratification (seed consistency, no leakage)
  - Model audit flags (distinguishing custom weights, COCO fallback, random weights)
  - Separation of heuristic scores from learned model metrics
"""

import os
import unittest
import numpy as np
from pathlib import Path

# Add backend directory to sys.path
import sys
backend_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(backend_dir))

from evaluate_models import compute_metrics, stratified_split, load_yolo, load_cnn_lstm
from ai_models import AIModelEngine


class TestEvaluationPipeline(unittest.TestCase):

    def test_metric_formulae_mathematical_precision(self):
        """Verify accuracy, precision, recall, and F1 calculations match official formulas."""
        # 10 ground truth: 6 fire (1), 4 no_fire (0)
        y_true = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        # Predictions:
        # TP = 4 (true fires predicted as fire)
        # FN = 2 (true fires predicted as normal)
        # FP = 1 (normal predicted as fire)
        # TN = 3 (normal predicted as normal)
        y_pred = np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 0])

        metrics = compute_metrics(y_true, y_pred, model_name="TestModel")

        # TP = 4, FP = 1, FN = 2, TN = 3
        expected_accuracy = (4 + 3) / 10.0   # 0.70
        expected_precision = 4 / (4 + 1)     # 0.80
        expected_recall = 4 / (4 + 2)        # 0.6667
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)  # ~0.7273

        self.assertAlmostEqual(metrics["accuracy"], expected_accuracy, places=4)
        self.assertAlmostEqual(metrics["precision_fire"], expected_precision, places=4)
        self.assertAlmostEqual(metrics["recall_fire"], expected_recall, places=4)
        self.assertAlmostEqual(metrics["f1_fire"], expected_f1, places=4)

        # Confusion matrix checks: [[TN, FP], [FN, TP]]
        cm = metrics["confusion_matrix"]["matrix"]
        self.assertEqual(cm[0][0], 3)  # TN
        self.assertEqual(cm[0][1], 1)  # FP
        self.assertEqual(cm[1][0], 2)  # FN
        self.assertEqual(cm[1][1], 4)  # TP

    def test_stratified_split_no_leakage_and_reproducibility(self):
        """Verify train/val/test splits do not leak samples and are reproducible."""
        paths = [f"img_{i}.jpg" for i in range(20)]
        labels = [1]*10 + [0]*10

        train_1, val_1, test_1 = stratified_split(paths, labels, seed=42)
        train_2, val_2, test_2 = stratified_split(paths, labels, seed=42)

        # 1. Exact reproducibility with same seed
        self.assertEqual(train_1, train_2)
        self.assertEqual(val_1, val_2)
        self.assertEqual(test_1, test_2)

        # 2. No leakage across splits (disjoint sets)
        set_train = set(train_1)
        set_val = set(val_1)
        set_test = set(test_1)
        self.assertEqual(len(set_train & set_val), 0)
        self.assertEqual(len(set_train & set_test), 0)
        self.assertEqual(len(set_val & set_test), 0)
        self.assertEqual(len(set_train | set_val | set_test), 20)

    def test_yolo_coco_weights_identified_as_cannot_detect_fire(self):
        """Verify generic yolov8n.pt is audited as lacking fire classes."""
        yolo_path = str(backend_dir / "yolov8n.pt")
        if os.path.exists(yolo_path):
            model, status, has_fire = load_yolo(yolo_path)
            self.assertFalse(has_fire)
            self.assertEqual(status, "coco_generic_weights")

    def test_cnn_lstm_missing_weights_identified_as_random(self):
        """Verify missing checkpoint is audited as random weights."""
        model, status = load_cnn_lstm("non_existent_weights.pth", device="cpu")
        self.assertEqual(status, "random_weights")

    def test_ai_model_engine_status_audit_integrity(self):
        """Verify AIModelEngine status distinguishes custom vs fallback vs random weights."""
        engine = AIModelEngine()
        status = engine.get_status()

        self.assertIn("models", status)
        self.assertIn("yolov8", status["models"])
        self.assertIn("efficientnet", status["models"])
        self.assertIn("cnn_lstm", status["models"])
        self.assertIn("heuristic_evaluation", status)

        # YOLO with COCO weights
        yolo_stat = status["models"]["yolov8"]
        self.assertTrue(yolo_stat["available"])
        self.assertTrue(yolo_stat["pretrained_fallback"])
        self.assertFalse(yolo_stat["can_detect_fire"])

        # CNN-LSTM with missing weights
        lstm_stat = status["models"]["cnn_lstm"]
        self.assertTrue(lstm_stat["available"])
        self.assertTrue(lstm_stat["random_weights"])
        self.assertFalse(lstm_stat["can_detect_fire"])
        self.assertFalse(lstm_stat["ready"])

        # Heuristic clearly marked as rule-based
        self.assertIn("heuristic_evaluation", status)
        self.assertIn("note", status["heuristic_evaluation"])


if __name__ == "__main__":
    unittest.main()
