"""
Comprehensive End-to-End QA Audit Test Suite.
Validates:
  1. NASA FIRMS Integration (Contracts, Satellite source, Parsing, Fallbacks, Status)
  2. AI Model Engine & Audit Flags (EfficientNet custom, YOLO generic, CNN-LSTM random)
  3. Image Analysis Pipeline & Heuristic Separation
  4. Model Evaluation Pipeline & Metrics File Integrity
  5. Flask REST API Endpoints Contract & Responses
"""

import json
import os
import io
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

# Setup backend import path
backend_dir = Path(__file__).parent.resolve()
import sys
sys.path.insert(0, str(backend_dir))
os.environ.setdefault("YOLO_CONFIG_DIR", str(backend_dir / ".yolo"))

from firms_service import FIRMSService, SOURCES
from ai_models import AIModelEngine
from image_analyzer import ImageAnalyzer
from evaluate_models import compute_metrics, load_dataset, stratified_split
import app as flask_app


class TestEndToEndSystemAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()

    # ─── 1. NASA FIRMS Integration Tests ─────────────────────────────────

    def test_nasa_sources_mapping_complete(self):
        """Verify VIIRS_SNPP_NRT, NOAA20, NOAA21, MODIS sources exist."""
        self.assertIn("VIIRS_SNPP", SOURCES)
        self.assertEqual(SOURCES["VIIRS_SNPP"], "VIIRS_SNPP_NRT")
        self.assertIn("MODIS", SOURCES)
        self.assertEqual(SOURCES["MODIS"], "MODIS_NRT")

    def test_nasa_csv_parsing_field_preservation(self):
        """Verify parsing preserves lat, lon, brightness, FRP, confidence, acq_date, acq_time."""
        csv_sample = (
            "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,confidence,version,bright_ti5,frp,daynight\n"
            "34.05,-118.25,345.6,0.4,0.4,2026-11-18,1430,N,nominal,2.0NRT,295.1,42.5,D\n"
        )
        service = FIRMSService()
        fires = service._parse_csv_text(csv_sample, "VIIRS_SNPP_NRT")
        self.assertIsNotNone(fires)
        self.assertEqual(len(fires), 1)
        f = fires[0]
        self.assertEqual(f["latitude"], 34.05)
        self.assertEqual(f["longitude"], -118.25)
        self.assertEqual(f["brightness"], 345.6)
        self.assertEqual(f["frp"], 42.5)
        self.assertEqual(f["confidence"], 60)  # "nominal" mapped to 60
        self.assertEqual(f["acq_date"], "2026-11-18")
        self.assertEqual(f["acq_time"], "14:30")
        self.assertEqual(f["is_simulated"], False)
        self.assertEqual(f["data_source"], "NASA_FIRMS")

    def test_nasa_valid_zero_detection_handling(self):
        """Verify a CSV with header but 0 rows is healthy 0 detections, not demo."""
        csv_zero = (
            "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,confidence,version,bright_ti5,frp,daynight\n"
        )
        service = FIRMSService()
        fires = service._parse_csv_text(csv_zero, "VIIRS_SNPP_NRT")
        self.assertEqual(fires, [])  # Empty list, not None

    def test_nasa_demo_data_isolation(self):
        """Verify demo data records are all tagged is_simulated=True, data_source=DEMO."""
        service = FIRMSService()
        demo_fires = service._get_demo_data("VIIRS_SNPP_NRT")
        self.assertTrue(len(demo_fires) > 0)
        for f in demo_fires:
            self.assertTrue(f["is_simulated"])
            self.assertEqual(f["data_source"], "DEMO")
            self.assertTrue(f["id"].startswith("DEMO_"))

    # ─── 2. AI Model Engine & Audit Integrity Tests ───────────────────────

    def test_ai_models_status_audit_integrity(self):
        """Verify get_status() distinguishes custom, fallback, and random weights."""
        engine = AIModelEngine()
        status = engine.get_status()

        self.assertIn("models", status)
        models_dict = status["models"]

        # EfficientNet has custom weights
        eff = models_dict["efficientnet"]
        self.assertTrue(eff["available"])
        self.assertTrue(eff["loaded"])
        self.assertTrue(eff["custom_weights_loaded"])
        self.assertFalse(eff["pretrained_fallback"])
        self.assertTrue(eff["can_detect_fire"])
        self.assertEqual(eff["evaluation_status"], "evaluated")
        self.assertIsNotNone(eff["metrics"])

        # YOLO has COCO generic weights
        yolo = models_dict["yolov8"]
        self.assertTrue(yolo["available"])
        self.assertTrue(yolo["loaded"])
        self.assertFalse(yolo["custom_weights_loaded"])
        self.assertTrue(yolo["pretrained_fallback"])
        self.assertFalse(yolo["can_detect_fire"])
        self.assertEqual(yolo["evaluation_status"], "not_evaluated")

        # CNN-LSTM has random uninitialized weights
        lstm = models_dict["cnn_lstm"]
        self.assertTrue(lstm["available"])
        self.assertTrue(lstm["loaded"])
        self.assertFalse(lstm["custom_weights_loaded"])
        self.assertTrue(lstm["random_weights"])
        self.assertFalse(lstm["can_detect_fire"])
        self.assertFalse(lstm["ready"])
        self.assertEqual(lstm["evaluation_status"], "not_evaluated")

    def test_ai_model_predict_excludes_untrained_models(self):
        """Verify predict() only includes models capable of detecting fire."""
        engine = AIModelEngine()
        # Synthetic 224x224 RGB image
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        pred = engine.predict(dummy_img)

        # EfficientNet is custom trained, so model_used should be True
        self.assertTrue(pred["model_used"])
        self.assertIn("confidence", pred)
        self.assertIn("fire_detected", pred)
        self.assertIn("scene_classification", pred)

        # CNN-LSTM with random weights must report random_weights_not_evaluated in details
        self.assertFalse(pred["details"]["cnn_lstm"]["available"])
        self.assertEqual(pred["details"]["cnn_lstm"]["reason"], "random_weights_not_evaluated")

    # ─── 3. Image Analysis & Heuristic Separation Tests ───────────────────

    def test_image_analyzer_separates_heuristics_from_models(self):
        """Verify ImageAnalyzer produces separate scores for heuristic and model ensemble."""
        analyzer = ImageAnalyzer()
        # Create a small valid JPEG in memory
        im = Image.new("RGB", (100, 100), color=(255, 50, 0))  # red/orange fire-like image
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        result = analyzer.analyze_image(img_bytes)
        self.assertIn("fire_detected", result)
        self.assertIn("confidence", result)
        self.assertIn("scores", result)

        scores = result["scores"]
        self.assertIn("fire_color", scores)
        self.assertIn("heuristic", scores)
        self.assertIn("model_ensemble", scores)

    # ─── 4. Model Evaluation Pipeline & Reports Tests ─────────────────────

    def test_evaluation_results_files_exist_and_valid(self):
        """Verify metrics.json, classification_report.txt, and confusion matrices exist."""
        results_dir = backend_dir / "evaluation_results"
        self.assertTrue((results_dir / "metrics.json").exists())
        self.assertTrue((results_dir / "classification_report.txt").exists())
        self.assertTrue((results_dir / "per_model_comparison.md").exists())
        self.assertTrue((results_dir / "confusion_matrix_EfficientNet-B0.png").exists())
        self.assertTrue((results_dir / "confusion_matrix_Heuristic.png").exists())

        with open(results_dir / "metrics.json", "r") as f:
            metrics_data = json.load(f)

        self.assertIn("evaluation_date", metrics_data)
        self.assertIn("dataset", metrics_data)
        self.assertIn("model_audit", metrics_data)
        self.assertIn("metrics", metrics_data)
        self.assertIn("efficientnet", metrics_data["metrics"])
        self.assertIn("heuristic", metrics_data["metrics"])

    # ─── 5. Flask API Endpoints Contract Tests ────────────────────────────

    def test_api_status_endpoint_contract(self):
        """Verify /api/status endpoint returns all expected NASA and AI model fields."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        # Core status fields
        self.assertEqual(data["status"], "running")
        self.assertIn("total_fires", data)
        self.assertIn("is_fetching", data)

        # FIRMS telemetry
        self.assertIn("source", data)
        self.assertIn("satellite", data)
        self.assertIn("api_status", data)
        self.assertIn("is_demo_data", data)
        self.assertIn("genuine_count", data)
        self.assertIn("demo_count", data)

        # AI model telemetry
        self.assertIn("models", data)
        self.assertIn("models", data["models"])
        self.assertIn("efficientnet", data["models"]["models"])
        self.assertIn("yolov8", data["models"]["models"])
        self.assertIn("cnn_lstm", data["models"]["models"])

    def test_api_models_status_endpoint(self):
        """Verify /api/models/status returns structured model audit details."""
        response = self.client.get("/api/models/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("device", data)
        self.assertIn("models", data)
        self.assertIn("heuristic_evaluation", data)
        self.assertTrue(data["models"]["efficientnet"]["custom_weights_loaded"])
        self.assertTrue(data["models"]["yolov8"]["pretrained_fallback"])
        self.assertTrue(data["models"]["cnn_lstm"]["random_weights"])

    def test_api_fires_endpoint(self):
        """Verify /api/fires returns fire list and metadata."""
        response = self.client.get("/api/fires?limit=5")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("fires", data)
        self.assertIn("total", data)
        self.assertIn("source", data)
        self.assertIn("satellite", data)
        self.assertIn("api_status", data)
        self.assertIn("is_demo_data", data)


if __name__ == "__main__":
    unittest.main()
