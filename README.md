# Forest Fire Detection System

This project now includes end-to-end emergency alerting, live camera analysis, sensor ingestion, and cloud-ready event storage.

## Tech Stack

- Backend: Python (Flask/FastAPI compatible), common data-science libraries for model inference
- Models: YOLOv8 for detection, EfficientNet-based classifier for scene classification
- Frontend: React + Vite, Tailwind CSS (optional), Leaflet / Google Maps for mapping
- Realtime: WebSockets / SSE for live feed and updates
- Database: PostgreSQL (recommended) or SQLite for local/dev
- Storage: AWS S3 or GCP Cloud Storage (cloud-abstracted)
- Messaging/Notifications: Twilio for SMS, webhook relay for mobile/FCM
- Containerization & Deployment: Docker, Kubernetes (optional), CI/CD pipelines

## Project structure

- backend/
  - app.py (backend entrypoint)
  - requirements.txt
  - .venv/ (created by launcher)
  - uploads/ (event logs, image uploads)
  - models/ (YOLO/EfficientNet weights and model artifacts)
  - api/ (API route handlers)
  - services/ (alerting, cloud storage adapters, sensor ingestion)

- frontend/
  - package.json
  - src/ (React app source)
  - public/ (static assets)
  - .env (Vite env file for API keys)

- scripts/
  - start.bat / run scripts and VS Code tasks

- docs/ (optional documentation and design notes)

> Note: This is a recommended structure — some files or folders may be named differently in the repository. Adjust names when integrating.

## One-Click Run (Recommended)

Use either:

- VS Code Task: `🔥 Run Everything`
- Windows launcher: `start.bat`

Both launchers now automatically:

- create backend virtual environment (`backend/.venv`) if missing
- install/sync backend dependencies from `backend/requirements.txt`
- install/sync frontend dependencies from `frontend/package.json`
- create `.env` files from `.env.example` if missing
- start backend and frontend servers

## Implemented Features

- SMS alerts using Twilio API
- Email alerts for high/critical incidents
- Mobile notifications through webhook relay (for FCM or custom mobile backend)
- Dashboard toast alerts with dispatch latency tracking
- Live camera feed analysis every 5 seconds
- Fire detection overlay boxes on live video
- Google Maps API support (with Leaflet fallback)
- Sensor ingestion for temperature, smoke, and humidity
- Cloud storage abstraction for AWS S3 or GCP Cloud Storage
- Real-time processing pipeline with automatic alert dispatch
- Multi-class scene differentiation: Fire vs Sunlight, Smoke vs Fog
- Fire heatmap map mode and live detection feed with scene labels
- Sensor time-series charts (temperature/smoke/risk vs time)
- Persistent storage for incidents, sensor readings, and detection logs
- Future prediction and pattern analysis endpoint

## Backend Setup

1. Install Python dependencies:

```bash
cd backend
pip install -r requirements.txt
```

2. Optional environment variables:

```env
# Cloud provider: local | aws | gcp
CLOUD_PROVIDER=local
CLOUD_BUCKET=firewatch-events
AWS_REGION=us-east-1

# Twilio SMS
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
TWILIO_TO_NUMBER=

# Mobile webhook relay
MOBILE_WEBHOOK_URL=
MOBILE_WEBHOOK_API_KEY=

# AI Model Settings
ENABLE_YOLOV8=true
YOLOV8_MODEL_PATH=yolov8n.pt
YOLO_FIRE_CLASS_NAMES=fire,smoke
YOLO_CONF_THRESHOLD=0.25
SCENE_CLASSES=fire,sunlight,smoke,fog,normal

ENABLE_EFFICIENTNET=true
# Optional: path to fine-tuned binary fire classifier checkpoint
EFFICIENTNET_WEIGHTS_PATH=
```

3. Run backend:

```bash
cd backend
py app.py
```

## Frontend Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Add Google Maps key in `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:5000/api
VITE_GOOGLE_MAPS_API_KEY=your_key_here
```

3. Run frontend:

```bash
cd frontend
npm run dev
```

## New API Endpoints

- `POST /api/alerts/sms` and `DELETE /api/alerts/sms`
- `POST /api/alerts/mobile` and `DELETE /api/alerts/mobile`
- `POST /api/live-feed/analyze`
- `POST /api/sensors` and `GET /api/sensors`
- `GET /api/cloud/status`
- `GET /api/models/status`
- `GET /api/sensors/history`
- `GET /api/incidents`
- `GET /api/detections/history`
- `GET /api/predictions`

## Notes

- Alert dispatch now tracks latency and marks whether delivery occurred within 5 seconds.
- Cloud event storage defaults to local files if AWS/GCP credentials are unavailable.
- Existing dashboard, image upload, region analytics, and history pages remain functional.
- YOLOv8 and EfficientNet are now part of image analysis. For best fire detection accuracy, provide custom fire/smoke model weights.
- Historical logs are stored in `backend/uploads/event_logs/*.jsonl` for analysis workflows.
