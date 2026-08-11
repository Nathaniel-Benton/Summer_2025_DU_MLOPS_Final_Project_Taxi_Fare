# Taxi Fare Prediction — MLOps Final Project

Predicts NYC yellow taxi trip fares from trip distance, passenger count, and
pickup/dropoff location IDs. Built for the COMP 4450 final project, covering
the MLOps lifecycle from experiment tracking through deployment.

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
pip install fastapi uvicorn pandas scikit-learn xgboost joblib wandb boto3
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

## Running the API

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`, with interactive
Swagger docs at `http://127.0.0.1:8000/docs`.

On startup you should see:
```
Model loaded successfully from W&B Artifacts!
Connected to DynamoDB table 'taxi-fare-predictions'
```

## API Reference

### `GET /health`
Returns whether the API is running and the model loaded successfully.

```bash
curl http://127.0.0.1:8000/health
```
```json
{"status": "ok", "message": "API is running and model is loaded"}
```

### `POST /predict`
Predicts the fare for a trip and logs the request to DynamoDB.

```bash
curl -X POST http://127.0.0.1:8000/predict \
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
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "actual_fare": 19.00
  }'
```
```json
{"status": "feedback recorded", "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}
```

## Model Training

Each training script pulls NYC TLC yellow taxi trip data, trains a model,
evaluates it, and logs everything to W&B:

```bash
python3 train_model_xgb_v2.2.py
```

Current production model: **XGBoost**, tuned via `RandomizedSearchCV`
(`train_model_xgb_v2.2.py`), trained on a 500k-row sample. See the W&B
project dashboard for full metric comparisons across all RF/XGB versions.
