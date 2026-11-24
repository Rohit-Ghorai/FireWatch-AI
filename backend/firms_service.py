"""
NASA FIRMS Data Fetching Service
Fetches near real-time active fire data from NASA's FIRMS API.
Supports VIIRS S-NPP, VIIRS NOAA-20, VIIRS NOAA-21, and MODIS satellite sources.
Includes robust error handling, caching, zero-detection handling, and strict demo labeling.
"""

import os
import requests
import time
import io
import csv
from datetime import datetime, UTC

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# NASA FIRMS API configuration
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Satellite sources mapping
SOURCES = {
    "VIIRS_SNPP": "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20": "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21": "VIIRS_NOAA21_NRT",
    "MODIS": "MODIS_NRT",
}

# Regional bounding boxes for queries (NASA area API limits queries to <= 100x100 degrees)
REGIONAL_BBOXES = [
    ("-125,24,-66,50",   "North America"),   # Continental US
    ("-70,-35,-35,5",    "South America"),    # Brazil / Amazon
    ("10,-15,40,10",     "Central Africa"),   # Central Africa
    ("95,5,115,25",      "Southeast Asia"),   # SE Asia
    ("115,-40,155,-10",  "Australia"),        # Australia
    ("70,5,90,30",       "India"),            # India
    ("60,50,140,70",     "Siberia"),          # Siberia
    ("15,34,40,45",      "Mediterranean"),    # Mediterranean
]

# In-memory cache for API queries
_cache = {}
_cache_ttl = 300  # 5 minutes


try:
    import pandas as pd
except ImportError:
    pd = None


def _is_na(val):
    if pd is not None:
        return pd.isna(val)
    return val is None or val == ""


class FIRMSFetchResult:
    """Encapsulates the result of a single NASA FIRMS API fetch attempt."""
    def __init__(self, success, fires, status, error=None):
        self.success = success
        self.fires = fires
        self.status = status  # "healthy", "invalid_key", "rate_limited", "error", "area_exceeded"
        self.error = error


class FIRMSService:
    def __init__(self, map_key=None, default_source=None):
        self._explicit_key = map_key
        self._default_source = default_source
        self.last_fetch_time = None
        self.last_successful_fetch = None
        self.api_status = "idle"
        self.api_error_message = None
        self.data_source = "DEMO"
        self.is_demo_data = True
        self.genuine_count = 0
        self.demo_count = 0
        self.fire_data = []

    @property
    def map_key(self):
        """Read map_key dynamically from env if not explicitly passed."""
        if self._explicit_key:
            return self._explicit_key
        return os.getenv("FIRMS_MAP_KEY", "").strip()

    @property
    def satellite_source(self):
        """Read satellite source dynamically from env, default to VIIRS_SNPP."""
        if self._default_source:
            return self._default_source
        env_source = os.getenv("FIRMS_SOURCE", os.getenv("FIRMS_SATELLITE_SOURCE", "VIIRS_SNPP")).strip()
        # Allow either friendly key ("VIIRS_SNPP") or raw API key ("VIIRS_SNPP_NRT")
        if env_source in SOURCES:
            return env_source
        for k, v in SOURCES.items():
            if v == env_source:
                return k
        return "VIIRS_SNPP"

    def _resolve_source_id(self, source):
        """Resolve friendly source name or return raw source string."""
        if not source:
            source = self.satellite_source
        return SOURCES.get(source, source)

    def fetch_fires(self, bbox="-180,-90,180,90", days=1, source=None):
        """
        Fetch active fire data from NASA FIRMS.
        
        Args:
            bbox: Bounding box as "west,south,east,north"
            days: Number of days to query (1-5)
            source: Satellite source key
        
        Returns:
            List of fire records as dicts
        """
        source_key = source or self.satellite_source
        source_id = self._resolve_source_id(source_key)
        cache_key = f"{source_id}_{bbox}_{days}"

        # 1. Check Cache
        if cache_key in _cache:
            entry = _cache[cache_key]
            cached_time = entry["time"]
            if time.time() - cached_time < _cache_ttl:
                # Cache hit: serve cached data without changing last_fetch_time or last_successful_fetch
                self.fire_data = entry["fires"]
                self.data_source = entry["data_source"]
                self.is_demo_data = entry["is_demo_data"]
                self.api_status = entry["api_status"]
                self.api_error_message = entry.get("error")
                self.genuine_count = entry.get("genuine_count", 0)
                self.demo_count = entry.get("demo_count", 0)
                # Keep original fetch times
                if entry.get("last_fetch_time"):
                    self.last_fetch_time = entry["last_fetch_time"]
                if entry.get("last_successful_fetch"):
                    self.last_successful_fetch = entry["last_successful_fetch"]
                print(f"[FIRMS] Returning cached data ({len(self.fire_data)} records, source: {self.data_source}).")
                return self.fire_data

        # 2. Perform Fetch
        is_global = (bbox.strip() == "-180,-90,180,90")
        if is_global:
            fetch_result = self._fetch_regional_queries(days, source_id)
        else:
            fetch_result = self._try_fetch_from_api(bbox, days, source_id)

        now_iso = datetime.now(UTC).isoformat()
        self.last_fetch_time = now_iso

        # 3. Handle Successful Fetch (Real NASA Data, either with detections or 0 detections)
        if fetch_result.success:
            self.fire_data = fetch_result.fires
            self.data_source = "NASA_FIRMS"
            self.is_demo_data = False
            self.genuine_count = len(fetch_result.fires)
            self.demo_count = 0
            self.api_status = "healthy"
            self.api_error_message = None
            self.last_successful_fetch = now_iso

            _cache[cache_key] = {
                "time": time.time(),
                "fires": self.fire_data,
                "data_source": self.data_source,
                "is_demo_data": self.is_demo_data,
                "api_status": self.api_status,
                "error": None,
                "genuine_count": self.genuine_count,
                "demo_count": 0,
                "last_fetch_time": self.last_fetch_time,
                "last_successful_fetch": self.last_successful_fetch,
            }
            print(f"[FIRMS] NASA API fetch SUCCESS: {len(self.fire_data)} genuine detections.")
            return self.fire_data

        # 4. Handle API Fetch Failure
        self.api_status = fetch_result.status
        self.api_error_message = fetch_result.error
        print(f"[FIRMS] NASA API fetch FAILED: {fetch_result.status} - {fetch_result.error}")

        # Check if we already have genuine NASA data from a previous fetch
        has_prior_genuine_data = (
            not self.is_demo_data and
            self.last_successful_fetch is not None and
            len(self.fire_data) >= 0
        )

        if has_prior_genuine_data:
            # PRESERVE genuine NASA data! Do NOT overwrite with demo data!
            self.api_status = "degraded"
            print(f"[FIRMS] Preserving prior genuine NASA dataset ({len(self.fire_data)} records).")
            # Update cache to avoid hammering API during failure
            _cache[cache_key] = {
                "time": time.time(),
                "fires": self.fire_data,
                "data_source": self.data_source,
                "is_demo_data": False,
                "api_status": self.api_status,
                "error": self.api_error_message,
                "genuine_count": self.genuine_count,
                "demo_count": 0,
                "last_fetch_time": self.last_fetch_time,
                "last_successful_fetch": self.last_successful_fetch,
            }
            return self.fire_data

        # 5. Fallback to Demo Data (only when no prior genuine data exists)
        print("[FIRMS] No prior genuine data available. Generating labeled SIMULATED demo data.")
        demo_fires = self._get_demo_data(source_id=source_id)
        self.fire_data = demo_fires
        self.data_source = "DEMO"
        self.is_demo_data = True
        self.genuine_count = 0
        self.demo_count = len(demo_fires)
        # Keep the specific failure reason if invalid_key, rate_limited, or error, else mark simulated
        if self.api_status not in ("invalid_key", "rate_limited", "error"):
            self.api_status = "simulated"

        _cache[cache_key] = {
            "time": time.time(),
            "fires": self.fire_data,
            "data_source": self.data_source,
            "is_demo_data": True,
            "api_status": self.api_status,
            "error": self.api_error_message,
            "genuine_count": 0,
            "demo_count": len(demo_fires),
            "last_fetch_time": self.last_fetch_time,
            "last_successful_fetch": self.last_successful_fetch,
        }
        return self.fire_data

    def _try_fetch_from_api(self, bbox, days, source_id):
        """
        Attempt a single FIRMS API fetch.
        Returns FIRMSFetchResult.
        """
        key = self.map_key
        if not key or key == "DEMO_KEY":
            return FIRMSFetchResult(
                success=False,
                fires=[],
                status="invalid_key",
                error="FIRMS_MAP_KEY is missing or set to DEMO_KEY. Register for a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/",
            )

        url = f"{FIRMS_BASE_URL}/{key}/{source_id}/{bbox}/{days}"

        try:
            # Mask key in logs
            masked_key = f"{key[:4]}...{key[-4:]}" if len(key) >= 8 else "****"
            masked_url = f"{FIRMS_BASE_URL}/{masked_key}/{source_id}/{bbox}/{days}"
            print(f"[FIRMS] Fetching: {masked_url}")

            headers = {
                "User-Agent": "FireWatch-AI/2.0 (Near-Real-Time Wildfire Intelligence)",
                "Accept": "text/csv, text/plain, */*",
            }
            response = requests.get(url, headers=headers, timeout=30)
            text = response.text.strip() if response.text else ""

            # Check HTTP Status Codes & Error text
            if response.status_code == 429 or "rate limit" in text.lower() or "exceeded maximum" in text.lower():
                return FIRMSFetchResult(
                    success=False,
                    fires=[],
                    status="rate_limited",
                    error="NASA FIRMS rate limit exceeded. Please wait before refreshing.",
                )

            if response.status_code in (401, 403) or "invalid map_key" in text.lower():
                return FIRMSFetchResult(
                    success=False,
                    fires=[],
                    status="invalid_key",
                    error="NASA FIRMS MAP_KEY is invalid or rejected by NASA.",
                )

            if response.status_code >= 500:
                return FIRMSFetchResult(
                    success=False,
                    fires=[],
                    status="error",
                    error=f"NASA FIRMS service error (HTTP {response.status_code}).",
                )

            if response.status_code != 200:
                return FIRMSFetchResult(
                    success=False,
                    fires=[],
                    status="error",
                    error=f"NASA FIRMS returned HTTP {response.status_code}: {text[:100]}",
                )

            # Response is HTTP 200: Validate CSV structure
            if not text:
                # Valid empty response
                return FIRMSFetchResult(success=True, fires=[], status="healthy")

            if text.startswith("<!") or text.startswith("<html") or text.startswith("{"):
                return FIRMSFetchResult(
                    success=False,
                    fires=[],
                    status="error",
                    error="NASA FIRMS returned HTML or non-CSV payload.",
                )

            if "area exceeds" in text.lower() or "coordinates exceeded" in text.lower():
                return FIRMSFetchResult(
                    success=False,
                    fires=[],
                    status="area_exceeded",
                    error="Bounding box exceeds NASA's 100x100 degree limit.",
                )

            # Parse CSV
            fires = self._parse_csv_text(text, source_id)
            if fires is None:
                # CSV parsing failed completely
                return FIRMSFetchResult(
                    success=False,
                    fires=[],
                    status="error",
                    error="Failed to parse CSV format returned from NASA FIRMS.",
                )

            return FIRMSFetchResult(success=True, fires=fires, status="healthy")

        except requests.exceptions.Timeout:
            return FIRMSFetchResult(
                success=False,
                fires=[],
                status="error",
                error="NASA FIRMS API request timed out (30s).",
            )
        except requests.exceptions.ConnectionError:
            return FIRMSFetchResult(
                success=False,
                fires=[],
                status="error",
                error="Network connection error reaching NASA FIRMS API.",
            )
        except Exception as e:
            return FIRMSFetchResult(
                success=False,
                fires=[],
                status="error",
                error=f"Unexpected error querying NASA FIRMS API: {str(e)}",
            )

    def _fetch_regional_queries(self, days, source_id):
        """
        Query regional bounding boxes to avoid NASA's 100x100 degree area limit.
        Merges results and tracks aggregate status.
        """
        all_fires = []
        any_success = False
        fatal_error = None
        fatal_status = None

        for bbox, region_name in REGIONAL_BBOXES:
            res = self._try_fetch_from_api(bbox, days, source_id)

            if res.success:
                any_success = True
                if res.fires:
                    print(f"[FIRMS] Got {len(res.fires)} fires from {region_name}")
                    all_fires.extend(res.fires)
            else:
                # If key is rejected or rate limited, don't keep hammering the rest of the regions
                if res.status in ("invalid_key", "rate_limited"):
                    return res

                fatal_status = res.status
                fatal_error = res.error

            # Polite pacing between regional calls
            time.sleep(0.15)

        if any_success:
            return FIRMSFetchResult(success=True, fires=all_fires, status="healthy")

        # If all regions failed
        return FIRMSFetchResult(
            success=False,
            fires=[],
            status=fatal_status or "error",
            error=fatal_error or "All regional FIRMS queries failed.",
        )

    def _parse_csv_text(self, text, source):
        """
        Parse CSV text into structured fire records.
        Returns list of fires (empty list if 0 detections), or None on format error.
        """
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if not lines:
            return []

        header_line = lines[0].lower()
        if "latitude" not in header_line or "longitude" not in header_line:
            print(f"[FIRMS] Missing latitude/longitude in CSV header: {header_line[:100]}")
            return None

        # If header exists and no data lines, it is a valid 0-detection response
        if len(lines) == 1:
            return []

        if pd is not None:
            return self._parse_with_pandas(text, source)
        else:
            return self._parse_manual_csv(text, source)

    def _parse_with_pandas(self, text, source):
        """Parse CSV using pandas."""
        try:
            df = pd.read_csv(io.StringIO(text))
            if df.empty:
                return []
            return self._process_dataframe(df, source)
        except Exception as e:
            print(f"[FIRMS] Pandas parse warning: {e}. Falling back to manual parser.")
            return self._parse_manual_csv(text, source)

    def _parse_manual_csv(self, text, source):
        """Fallback CSV parser using Python's standard csv library."""
        fires = []
        try:
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                try:
                    lat_val = row.get("latitude")
                    lng_val = row.get("longitude")
                    if lat_val is None or lng_val is None:
                        continue

                    lat = float(lat_val)
                    lng = float(lng_val)

                    # Brightness: check bright_ti4 (VIIRS), brightness (MODIS), or bright_ti5
                    raw_b = row.get("bright_ti4") or row.get("brightness") or row.get("bright_ti5") or 0.0
                    try:
                        brightness = round(float(raw_b), 2)
                    except (ValueError, TypeError):
                        brightness = 0.0

                    # FRP (Fire Radiative Power)
                    raw_frp = row.get("frp", 0.0)
                    try:
                        frp = round(float(raw_frp), 2)
                    except (ValueError, TypeError):
                        frp = 0.0

                    # Confidence: low/nominal/high or numeric
                    conf_raw = str(row.get("confidence", "nominal")).strip().lower()
                    if conf_raw in ("low", "l"):
                        confidence = 30
                    elif conf_raw in ("nominal", "n"):
                        confidence = 60
                    elif conf_raw in ("high", "h"):
                        confidence = 90
                    else:
                        try:
                            confidence = int(float(conf_raw))
                            confidence = max(0, min(100, confidence))
                        except Exception:
                            confidence = 50

                    acq_date = str(row.get("acq_date", "")).strip()
                    acq_time_raw = str(row.get("acq_time", "")).strip()
                    acq_time = acq_time_raw.zfill(4) if acq_time_raw.isdigit() else "0000"
                    daynight = str(row.get("daynight", "D")).strip().upper()
                    if daynight not in ("D", "N"):
                        daynight = "D"

                    severity = self._classify_severity(confidence, frp, brightness)

                    fires.append({
                        "id": f"{source}_{lat:.4f}_{lng:.4f}_{acq_date}_{acq_time}",
                        "latitude": lat,
                        "longitude": lng,
                        "brightness": brightness,
                        "frp": frp,
                        "confidence": confidence,
                        "acq_date": acq_date,
                        "acq_time": f"{acq_time[:2]}:{acq_time[2:]}",
                        "daynight": daynight,
                        "satellite": source,
                        "severity": severity,
                        "data_source": "NASA_FIRMS",
                        "is_simulated": False,
                    })
                except Exception:
                    continue
            return fires
        except Exception as e:
            print(f"[FIRMS] Manual CSV parser failed: {e}")
            return None

    def _process_dataframe(self, df, source):
        """Process raw FIRMS CSV DataFrame into structured fire records."""
        fires = []
        col_map = {col.lower(): col for col in df.columns}

        lat_col = col_map.get("latitude")
        lng_col = col_map.get("longitude")
        if not lat_col or not lng_col:
            return []

        b_col = col_map.get("bright_ti4") or col_map.get("brightness") or col_map.get("bright_ti5")
        frp_col = col_map.get("frp")
        conf_col = col_map.get("confidence")
        date_col = col_map.get("acq_date")
        time_col = col_map.get("acq_time")
        dn_col = col_map.get("daynight")

        for _, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                lng = float(row[lng_col])

                brightness = float(row[b_col]) if b_col and b_col in row and not pd.isna(row[b_col]) else 0.0
                frp = float(row[frp_col]) if frp_col and frp_col in row and not pd.isna(row[frp_col]) else 0.0

                conf_raw = row[conf_col] if conf_col and conf_col in row else "nominal"
                if isinstance(conf_raw, str):
                    c_lower = conf_raw.strip().lower()
                    if c_lower in ("low", "l"):
                        confidence = 30
                    elif c_lower in ("nominal", "n"):
                        confidence = 60
                    elif c_lower in ("high", "h"):
                        confidence = 90
                    else:
                        try:
                            confidence = int(float(c_lower))
                        except Exception:
                            confidence = 50
                else:
                    try:
                        confidence = int(float(conf_raw))
                    except Exception:
                        confidence = 50
                confidence = max(0, min(100, confidence))

                acq_date = str(row[date_col]).strip() if date_col and not pd.isna(row[date_col]) else ""
                acq_time_raw = str(row[time_col]).strip() if time_col and not pd.isna(row[time_col]) else ""
                # Strip any decimal if parsed as float
                if "." in acq_time_raw:
                    acq_time_raw = acq_time_raw.split(".")[0]
                acq_time = acq_time_raw.zfill(4) if acq_time_raw.isdigit() else "0000"
                daynight = str(row[dn_col]).strip().upper() if dn_col and not pd.isna(row[dn_col]) else "D"
                if daynight not in ("D", "N"):
                    daynight = "D"

                severity = self._classify_severity(confidence, frp, brightness)

                fires.append({
                    "id": f"{source}_{lat:.4f}_{lng:.4f}_{acq_date}_{acq_time}",
                    "latitude": lat,
                    "longitude": lng,
                    "brightness": round(brightness, 2),
                    "frp": round(frp, 2),
                    "confidence": confidence,
                    "acq_date": acq_date,
                    "acq_time": f"{acq_time[:2]}:{acq_time[2:]}",
                    "daynight": daynight,
                    "satellite": source,
                    "severity": severity,
                    "data_source": "NASA_FIRMS",
                    "is_simulated": False,
                })
            except Exception:
                continue

        return fires

    def _classify_severity(self, confidence, frp, brightness):
        """Classify fire severity based on multiple indicators."""
        score = 0
        score += min((confidence / 100.0) * 40.0, 40.0)
        score += min((frp / 100.0) * 30.0, 30.0)
        score += min((brightness / 500.0) * 30.0, 30.0)

        if score >= 70:
            return "critical"
        elif score >= 50:
            return "high"
        elif score >= 30:
            return "moderate"
        else:
            return "low"

    def _get_demo_data(self, source_id="VIIRS_SNPP_NRT"):
        """
        Generate clearly labeled simulated/demo fire data.
        Every record is explicitly tagged as DEMO and SIMULATED.
        """
        import random
        demo_fires = []
        regions = [
            {"name": "Amazon", "lat_range": (-15, -3), "lng_range": (-65, -45)},
            {"name": "Central Africa", "lat_range": (-10, 5), "lng_range": (15, 30)},
            {"name": "Southeast Asia", "lat_range": (10, 20), "lng_range": (95, 110)},
            {"name": "Australia", "lat_range": (-35, -20), "lng_range": (120, 150)},
            {"name": "Western US", "lat_range": (33, 45), "lng_range": (-122, -110)},
            {"name": "Siberia", "lat_range": (55, 65), "lng_range": (90, 130)},
            {"name": "Mediterranean", "lat_range": (35, 42), "lng_range": (20, 35)},
            {"name": "India", "lat_range": (15, 28), "lng_range": (75, 85)},
        ]

        now_utc = datetime.now(UTC)
        today_str = now_utc.strftime("%Y-%m-%d")

        for region in regions:
            num_fires = random.randint(5, 20)
            for i in range(num_fires):
                lat = round(random.uniform(*region["lat_range"]), 4)
                lng = round(random.uniform(*region["lng_range"]), 4)
                confidence = random.randint(30, 95)
                frp = round(random.uniform(5.0, 180.0), 1)
                brightness = round(random.uniform(310.0, 480.0), 1)
                severity = self._classify_severity(confidence, frp, brightness)

                demo_fires.append({
                    "id": f"DEMO_{region['name']}_{i}_{today_str}",
                    "latitude": lat,
                    "longitude": lng,
                    "brightness": brightness,
                    "frp": frp,
                    "confidence": confidence,
                    "acq_date": today_str,
                    "acq_time": f"{random.randint(0,23):02d}:{random.randint(0,59):02d}",
                    "daynight": random.choice(["D", "N"]),
                    "satellite": f"DEMO ({source_id})",
                    "severity": severity,
                    "data_source": "DEMO",
                    "is_simulated": True,
                })

        return demo_fires

    def get_status(self):
        """Return comprehensive FIRMS service metadata."""
        return {
            "source": self.data_source,
            "satellite": self.satellite_source,
            "satellite_id": self._resolve_source_id(self.satellite_source),
            "api_status": self.api_status,
            "last_successful_fetch": self.last_successful_fetch,
            "last_updated": self.last_fetch_time,
            "record_count": len(self.fire_data),
            "genuine_count": self.genuine_count,
            "demo_count": self.demo_count,
            "is_demo_data": self.is_demo_data,
            "refresh_interval": _cache_ttl,
            "has_map_key": bool(self.map_key and self.map_key != "DEMO_KEY"),
            "error": self.api_error_message,
        }

    def get_fires_by_region(self, region_bbox, days=1):
        """Fetch fires for a specific region bounding box."""
        return self.fetch_fires(bbox=region_bbox, days=days)

    def get_all_sources(self, bbox="-180,-90,180,90", days=1):
        """Fetch fires from all satellite sources and merge."""
        all_fires = []
        for source_key in SOURCES:
            fires = self.fetch_fires(bbox=bbox, days=days, source=source_key)
            all_fires.extend(fires)
        return all_fires
