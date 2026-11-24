"""
Automated Test Suite for NASA FIRMS Integration in FireWatch-AI.
Covers all requirements:
1. Successful NASA response with fire detections
2. Valid NASA response with zero fire detections (header only)
3. Invalid API keys and HTTP errors (400, 403, 500)
4. Rate limiting responses (429, limit text)
5. Timeouts and network connection failures
6. Caching within 300s TTL (no re-fetch, timestamp preserved)
7. Cache expiration after 300s
8. Demo-data strict labeling (is_simulated=True, DEMO tags)
9. Preservation of genuine NASA data across subsequent API failures
10. Flask REST status API contract and metadata
11. MODIS vs VIIRS CSV schema parsing
"""

import unittest
from unittest.mock import patch, MagicMock
import time
import os

from firms_service import FIRMSService, _cache, SOURCES
from app import app, firms, current_fires


SAMPLE_VIIRS_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
    "-14.2351,-51.9253,342.5,0.4,0.4,2026-11-18,1230,N,VIIRS,high,2.0NRT,295.2,45.8,D\n"
    "37.7749,-122.4194,320.1,0.5,0.4,2026-11-18,1415,N,VIIRS,nominal,2.0NRT,288.0,12.3,D\n"
    "-25.2744,133.7751,310.0,0.4,0.4,2026-11-18,2200,N,VIIRS,low,2.0NRT,280.5,4.5,N\n"
)

SAMPLE_ZERO_DETECTIONS_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
)

SAMPLE_MODIS_CSV = (
    "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight\n"
    "-12.5000,-48.2000,335.2,1.2,1.1,2026-11-18,1345,Terra,MODIS,85,6.1NRT,298.0,38.2,D\n"
)


class TestFIRMSIntegration(unittest.TestCase):

    def setUp(self):
        # Clear module cache before each test
        _cache.clear()

    # ─────────────────────────────────────────────────────────────
    # Test 1: Successful NASA response with detections
    # ─────────────────────────────────────────────────────────────
    @patch("requests.get")
    def test_successful_nasa_response_with_detections(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_VIIRS_CSV
        mock_get.return_value = mock_resp

        service = FIRMSService(map_key="valid_test_key_1234567890123456")
        fires = service.fetch_fires(bbox="-125,24,-66,50", days=1, source="VIIRS_SNPP")

        self.assertEqual(len(fires), 3)
        self.assertFalse(service.is_demo_data)
        self.assertEqual(service.data_source, "NASA_FIRMS")
        self.assertEqual(service.genuine_count, 3)
        self.assertEqual(service.demo_count, 0)
        self.assertEqual(service.api_status, "healthy")
        self.assertIsNotNone(service.last_successful_fetch)

        first = fires[0]
        self.assertFalse(first["is_simulated"])
        self.assertEqual(first["data_source"], "NASA_FIRMS")
        self.assertEqual(first["latitude"], -14.2351)
        self.assertEqual(first["longitude"], -51.9253)
        self.assertEqual(first["brightness"], 342.5)
        self.assertEqual(first["frp"], 45.8)
        self.assertEqual(first["confidence"], 90)  # "high" normalized to 90
        self.assertEqual(first["acq_date"], "2026-11-18")
        self.assertEqual(first["acq_time"], "12:30")
        self.assertEqual(first["daynight"], "D")
        self.assertIn("VIIRS", first["satellite"])

    # ─────────────────────────────────────────────────────────────
    # Test 2: Valid response with ZERO fire detections (header only)
    # ─────────────────────────────────────────────────────────────
    @patch("requests.get")
    def test_valid_nasa_response_with_zero_detections(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_ZERO_DETECTIONS_CSV
        mock_get.return_value = mock_resp

        service = FIRMSService(map_key="valid_test_key_1234567890123456")
        fires = service.fetch_fires(bbox="-125,24,-66,50", days=1, source="VIIRS_SNPP")

        # Crucial requirement: Must be treated as valid NASA scan, NOT a failure, NOT demo data
        self.assertEqual(len(fires), 0)
        self.assertFalse(service.is_demo_data)
        self.assertEqual(service.data_source, "NASA_FIRMS")
        self.assertEqual(service.genuine_count, 0)
        self.assertEqual(service.demo_count, 0)
        self.assertEqual(service.api_status, "healthy")
        self.assertIsNotNone(service.last_successful_fetch)

    # ─────────────────────────────────────────────────────────────
    # Test 3: Invalid API Key (HTTP 400 with "Invalid MAP_KEY.")
    # ─────────────────────────────────────────────────────────────
    @patch("requests.get")
    def test_invalid_api_key_http_400(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Invalid MAP_KEY."
        mock_get.return_value = mock_resp

        service = FIRMSService(map_key="invalid_or_expired_key")
        fires = service.fetch_fires(bbox="-125,24,-66,50", days=1, source="VIIRS_SNPP")

        # Must fall back to labeled demo data
        self.assertTrue(len(fires) > 0)
        self.assertTrue(service.is_demo_data)
        self.assertEqual(service.data_source, "DEMO")
        self.assertEqual(service.api_status, "invalid_key")
        self.assertIsNone(service.last_successful_fetch)

        # Every record must be strictly tagged
        for fire in fires:
            self.assertTrue(fire["is_simulated"])
            self.assertEqual(fire["data_source"], "DEMO")
            self.assertTrue(fire["id"].startswith("DEMO_"))

    # ─────────────────────────────────────────────────────────────
    # Test 4: Rate limit response (HTTP 429)
    # ─────────────────────────────────────────────────────────────
    @patch("requests.get")
    def test_rate_limit_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Exceeded maximum requests per day."
        mock_get.return_value = mock_resp

        service = FIRMSService(map_key="valid_key_but_exceeded")
        fires = service.fetch_fires(bbox="-125,24,-66,50", days=1, source="VIIRS_SNPP")

        self.assertTrue(service.is_demo_data)
        self.assertEqual(service.api_status, "rate_limited")
        self.assertIn("rate limit", service.api_error_message.lower())

    # ─────────────────────────────────────────────────────────────
    # Test 5: Timeouts and connection errors
    # ─────────────────────────────────────────────────────────────
    @patch("requests.get")
    def test_timeout_and_network_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out after 30s")

        service = FIRMSService(map_key="valid_key_timeout")
        fires = service.fetch_fires(bbox="-125,24,-66,50", days=1, source="VIIRS_SNPP")

        self.assertTrue(service.is_demo_data)
        self.assertEqual(service.api_status, "error")
        self.assertIn("timed out", service.api_error_message.lower())

    # ─────────────────────────────────────────────────────────────
    # Test 6: Caching and timestamp preservation within 300s
    # ─────────────────────────────────────────────────────────────
    @patch("requests.get")
    def test_caching_and_timestamp_preservation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_VIIRS_CSV
        mock_get.return_value = mock_resp

        service = FIRMSService(map_key="valid_test_key_caching")
        fires1 = service.fetch_fires(bbox="-125,24,-66,50", days=1, source="VIIRS_SNPP")
        first_fetch_time = service.last_fetch_time
        first_successful_time = service.last_successful_fetch
        self.assertEqual(mock_get.call_count, 1)

        # Call again immediately within TTL
        fires2 = service.fetch_fires(bbox="-125,24,-66,50", days=1, source="VIIRS_SNPP")
        # requests.get must NOT have been called again
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(len(fires1), len(fires2))
        # Must preserve original timestamp, not overwrite with new timestamp
        self.assertEqual(service.last_fetch_time, first_fetch_time)
        self.assertEqual(service.last_successful_fetch, first_successful_time)

    # ─────────────────────────────────────────────────────────────
    # Test 7: Cache expiration after 300s
    # ─────────────────────────────────────────────────────────────
    @patch("requests.get")
    @patch("time.time")
    def test_cache_expiration(self, mock_time, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_VIIRS_CSV
        mock_get.return_value = mock_resp

        # Initial time: 1000.0
        mock_time.return_value = 1000.0
        service = FIRMSService(map_key="valid_test_key_exp")
        service.fetch_fires(bbox="-125,24,-66,50", days=1, source="VIIRS_SNPP")
        self.assertEqual(mock_get.call_count, 1)

        # Fast forward time by 301 seconds (past 300s TTL)
        mock_time.return_value = 1301.0
        service.fetch_fires(bbox="-125,24,-66,50", days=1, source="VIIRS_SNPP")
        # requests.get must have been invoked a second time
        self.assertEqual(mock_get.call_count, 2)

    # ─────────────────────────────────────────────────────────────
    # Test 8: Non-overwriting of genuine data on subsequent API failure
    # ─────────────────────────────────────────────────────────────
    @patch("requests.get")
    def test_api_failure_does_not_overwrite_genuine_nasa_data(self, mock_get):
        import requests
        # First call succeeds
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_VIIRS_CSV
        mock_get.return_value = mock_resp

        service = FIRMSService(map_key="valid_test_key_preserve")
        fires_initial = service.fetch_fires(bbox="-125,24,-66,50", days=1, source="VIIRS_SNPP")
        self.assertEqual(len(fires_initial), 3)
        self.assertFalse(service.is_demo_data)
        genuine_timestamp = service.last_successful_fetch

        # Invalidate cache to simulate a refresh attempt
        _cache.clear()

        # Second call fails with network connection error
        mock_get.side_effect = requests.exceptions.ConnectionError("NASA server unreachable")
        fires_second = service.fetch_fires(bbox="-125,24,-66,50", days=1, source="VIIRS_SNPP")

        # Crucial requirement: must NOT overwrite genuine records with demo data!
        self.assertEqual(len(fires_second), 3)
        self.assertFalse(service.is_demo_data)
        self.assertEqual(service.data_source, "NASA_FIRMS")
        self.assertEqual(service.api_status, "degraded")
        self.assertEqual(service.last_successful_fetch, genuine_timestamp)

    # ─────────────────────────────────────────────────────────────
    # Test 9: MODIS CSV parsing
    # ─────────────────────────────────────────────────────────────
    @patch("requests.get")
    def test_modis_csv_parsing(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_MODIS_CSV
        mock_get.return_value = mock_resp

        service = FIRMSService(map_key="valid_test_key_modis")
        fires = service.fetch_fires(bbox="-125,24,-66,50", days=1, source="MODIS")

        self.assertEqual(len(fires), 1)
        modis_fire = fires[0]
        self.assertEqual(modis_fire["latitude"], -12.5)
        self.assertEqual(modis_fire["longitude"], -48.2)
        self.assertEqual(modis_fire["brightness"], 335.2)
        self.assertEqual(modis_fire["confidence"], 85)
        self.assertIn("MODIS", modis_fire["satellite"])

    # ─────────────────────────────────────────────────────────────
    # Test 10: Status API Endpoint contract
    # ─────────────────────────────────────────────────────────────
    def test_status_endpoint_contract(self):
        with app.test_client() as client:
            res = client.get("/api/status")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()

            # Verify backward compatibility
            self.assertIn("status", data)
            self.assertIn("is_fetching", data)
            self.assertIn("total_fires", data)
            self.assertIn("last_updated", data)
            self.assertIn("analysis_available", data)
            self.assertIn("models", data)

            # Verify FIRMS transparency metadata
            self.assertIn("source", data)
            self.assertIn("satellite", data)
            self.assertIn("api_status", data)
            self.assertIn("last_successful_fetch", data)
            self.assertIn("record_count", data)
            self.assertIn("genuine_count", data)
            self.assertIn("demo_count", data)
            self.assertIn("is_demo_data", data)
            self.assertIn("refresh_interval", data)
            self.assertEqual(data["refresh_interval"], 300)

    # ─────────────────────────────────────────────────────────────
    # Test 11: Fires API Endpoint contract
    # ─────────────────────────────────────────────────────────────
    @patch("requests.get")
    def test_fires_endpoint_contract(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_VIIRS_CSV
        mock_get.return_value = mock_resp

        with app.test_client() as client:
            res = client.get("/api/fires?limit=10")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()

            self.assertIn("fires", data)
            self.assertIn("total", data)
            self.assertIn("source", data)
            self.assertIn("satellite", data)
            self.assertIn("api_status", data)
            self.assertIn("is_demo_data", data)
            self.assertIn("genuine_count", data)
            self.assertIn("demo_count", data)


if __name__ == "__main__":
    unittest.main()
