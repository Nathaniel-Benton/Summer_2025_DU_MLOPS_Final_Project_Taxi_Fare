# Taxi Fare Prediction — MLOps Final Project

Predicts NYC yellow taxi trip fares from trip distance, passenger count, and
pickup/dropoff location IDs. Built for the COMP 4450 final project, covering
the MLOps lifecycle from experiment tracking through deployment.

## Project Status

| Phase | Component | Status |
|---|---|---|
| 1 | Experiment tracking (W&B) | Done |
| 1 | Model registry / versioning | Done (RF + XGBoost variants tracked) |
| 2.1 | FastAPI backend (`/health`, `/predict`) | Done |
| 2.2 | Cloud database (DynamoDB) + prediction logging | Done |
| 2.2 | Feedback endpoint for live accuracy | Done |
| 3.1 | Frontend (Streamlit prediction app) | Done |
| 3.2 | Monitoring dashboard | Done |
| 4 | Unit / integration tests | Not started |
| 4.2 | CI/CD (GitHub Actions) | Not started |
| 5 | Docker containerization | Not started |
| 5 | AWS EC2 deployment | Not started |

## Architecture

- **Model training**: `train_model_rf_v1.*.py` (Random Forest) and
  `train_model_xgb_v2.*.py` (XGBoost) — each logs hyperparameters, metrics
  (RMSE/MAE/R²), and the trained model artifact to Weights & Biases.
- **Serving model**: The tuned XGBoost regressor
  (`taxi-fare-xgboost-tuned:latest` in W&B) is what `main.py` loads and
  serves.
- **Backend**: `main.py` — FastAPI app that loads the model from W&B on
  startup, serves predictions, and logs every prediction to DynamoDB.
- **Data store**: DynamoDB table `taxi-fare-predictions`, on-demand billing
  mode. Partition key is `prediction_id` (String, UUID).
- **Prediction frontend**: `streamlit_app.py` — lets a user enter trip
  details, calls the backend's `/predict` endpoint, and lets them submit
  the actual observed fare afterward via `/feedback`.
- **Monitoring dashboard**: `monitoring_dashboard.py` — a separate Streamlit
  app that reads directly from DynamoDB (not from the backend or any shared
  file) and visualizes latency over time, predicted fare distribution, and
  live accuracy computed from feedback. Designed to run as an independent
  app so it can be deployed to its own EC2 instance in Phase 5.

## Prerequisites

- Python 3.10+
- A Weights & Biases account with access to the
  `models-university-of-denver9526/DU_Summer25_Final_Project_Taxi_Fare`
  project
- An AWS account with a DynamoDB table named `taxi-fare-predictions`
  (partition key: `prediction_id`, type String, on-demand billing)

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd final-project
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Weights & Biases

```bash
wandb login
```
Follow the prompt to paste your API key (found under your W&B account
settings).

### 4. Configure AWS credentials

Create `~/.aws/credentials`:
```ini
[default]
aws_access_key_id=YOUR_ACCESS_KEY
aws_secret_access_key=YOUR_SECRET_KEY
aws_session_token=YOUR_SESSION_TOKEN
```
> The `aws_session_token` line is required if you're using AWS Academy
> Learner Lab temporary credentials. Omit it for a standard long-lived IAM
> user.

And `~/.aws/config`:
```ini
[default]
region=us-east-1
```
Use whichever region your DynamoDB table was actually created in.

Verify both are working:
```bash
aws sts get-caller-identity
python3 -c "import boto3; print(boto3.resource('dynamodb').Table('taxi-fare-predictions').table_status)"
```
Both should succeed without error (the second should print `ACTIVE`).

> **Note (AWS Academy Learner Lab users):** these credentials expire when
> your lab session ends. If you get an auth error after some time has
> passed, re-copy fresh credentials from the lab's "AWS Details" panel.

## Running the API

```bash
uvicorn main:app --reload --port 8001
```

> **Note:** Port 8000 is used here instead of the FastAPI default because it
> can be occupied on some dev machines by Docker Desktop / the WSL network
> relay (`wslrelay.exe` on Windows+WSL2 setups), even though nothing shows
> up in WSL's own `lsof`. If you see `[Errno 98] Address already in use` on
> 8000 and don't want to track down and kill whatever's holding it, just
> use 8001 instead as shown above.

The API will be available at `http://127.0.0.1:8001`, with interactive
Swagger docs at `http://127.0.0.1:8001/docs`.

On startup you should see:
```
Model loaded successfully from W&B Artifacts!
Connected to DynamoDB table 'taxi-fare-predictions'
```

## API Reference

### `GET /health`
Returns whether the API is running and the model loaded successfully.

```bash
curl http://127.0.0.1:8001/health
```
```json
{"status": "ok", "message": "API is running and model is loaded"}
```

### `POST /predict`
Predicts the fare for a trip and logs the request to DynamoDB.

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trip_distance": 3.5,
    "passenger_count": 1,
    "PULocationID": 142,
    "DOLocationID": 236,
    "RatecodeID": 1
  }'
```
```json
{
  "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "predicted_fare_amount": 18.42
}
```

### `POST /feedback`
Records the actual observed fare for a previous prediction, used to
calculate live accuracy on the monitoring dashboard (Phase 3.2).

```bash
curl -X POST http://127.0.0.1:8001/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "actual_fare": 19.00
  }'
```
```json
{"status": "feedback recorded", "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}
```

## DynamoDB Schema (`taxi-fare-predictions`)

| Attribute | Type | Notes |
|---|---|---|
| `prediction_id` | String (PK) | UUID, generated per request |
| `timestamp` | String | ISO 8601, UTC |
| `trip_distance` | Number | input feature |
| `passenger_count` | Number | input feature |
| `PULocationID` | Number | input feature |
| `DOLocationID` | Number | input feature |
| `RatecodeID` | Number | input feature |
| `predicted_fare` | Number | model output |
| `latency_ms` | Number | prediction latency, for monitoring |
| `model_version` | String | which model served the request |
| `feedback_fare` | Number | optional, added later via `/feedback` |

## Running the Prediction Frontend

With the backend already running (see above), in a second terminal:

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Enter trip details, get a predicted fare,
and optionally submit the actual fare afterward to feed the monitoring
dashboard's live accuracy metric.

By default it talks to the backend at `http://127.0.0.1:8001`. Override
with the `API_URL` env var if your backend is running elsewhere:
```bash
API_URL=http://127.0.0.1:8001 streamlit run streamlit_app.py
```

## Running the Monitoring Dashboard

This is a separate app that reads directly from DynamoDB — it does not
depend on the backend being up, but it does need an active AWS session.

```bash
streamlit run monitoring_dashboard.py --server.port 8502
```

Run on a different port than the prediction app so both can run
simultaneously. Shows prediction latency over time, distribution of
predicted fares, live accuracy (MAE / MAPE from feedback), and a raw log
table. Click **Refresh** to re-pull from DynamoDB (results are otherwise
cached for 30 seconds).

## Model Training

Each training script pulls NYC TLC yellow taxi trip data, trains a model,
evaluates it, and logs everything to W&B:

```bash
python3 train_model_xgb_v2.2.py
```

Current production model: **XGBoost**, tuned via `RandomizedSearchCV`
(`train_model_xgb_v2.2.py`), trained on a 500k-row sample. See the W&B
project dashboard for full metric comparisons across all RF/XGB versions.

## Known Limitations / TODO

- No caching layer yet for repeated identical requests (mentioned as
  optional in the assignment).
- No automated tests or CI/CD pipeline yet.
- Not yet containerized or deployed to EC2.
- Model promotion in W&B (staging/production aliasing) is inconsistent
  across versions — only `v1.0` explicitly tags aliases.
- Location IDs are entered as raw numbers in the prediction frontend, not
  matched to human-readable zone names.

## Cost Notes (AWS Academy Learner Lab)

- DynamoDB on-demand billing stays within the free tier at this project's
  scale — cost is negligible.
- EC2 instances (used in later phases) pause automatically when a lab
  session ends but must be manually restarted; they are the primary cost
  driver, not DynamoDB.
- Avoid RDS/NAT gateways unless required — they continue billing even
  outside active lab sessions.
  <!-- testing CI workflow -->