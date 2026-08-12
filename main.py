import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3
import joblib
import pandas as pd
import wandb
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app = FastAPI(title="Taxi Fare Prediction", version="2.2")

# --- Initialize W&B API and Load Production Model ---
# This downloads the best-performing model version directly from W&B
try:
    print("Connecting to W&B to download the latest production model...")
    api = wandb.Api()

    # Target the production-tuned XGBoost artifact we identified on the dashboard
    artifact_ref = (
        "models-university-of-denver9526/"
        "DU_Summer25_Final_Project_Taxi_Fare/"
        "taxi-fare-xgboost-tuned:latest"
    )
    artifact = api.artifact(artifact_ref)
    artifact_dir = artifact.download()

    # Load the model weights into memory
    model_path = f"{artifact_dir}/taxi_fare_xgb_tuned.pkl"
    model = joblib.load(model_path)
    print("Model loaded successfully from W&B Artifacts!")
except Exception as e:
    print(f"Error loading model from W&B: {e}")
    model = None

# Optional: Load dataset for example/random sampling endpoints if needed
try:
    df = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet")
except Exception:
    df = None

# --- Initialize DynamoDB connection ---
# Table used to log every prediction request for later monitoring/drift analysis
DYNAMODB_TABLE_NAME = "taxi-fare-predictions"
MODEL_VERSION = "xgb_tuned_latest"

try:
    dynamodb = boto3.resource("dynamodb")
    prediction_table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    # Cheap call to confirm the table is reachable at startup
    _ = prediction_table.table_status
    print(f"Connected to DynamoDB table '{DYNAMODB_TABLE_NAME}'")
except Exception as e:
    print(f"Warning: could not connect to DynamoDB table '{DYNAMODB_TABLE_NAME}': {e}")
    prediction_table = None


# Define input schema matching the model's training features
class TripInput(BaseModel):
    trip_distance: float
    passenger_count: int
    PULocationID: int
    DOLocationID: int
    RatecodeID: int


class FeedbackInput(BaseModel):
    prediction_id: str
    actual_fare: float


def build_input_dataframe(trip: TripInput) -> pd.DataFrame:
    """
    Convert a TripInput into the single-row DataFrame layout the model
    expects. Pulled out as its own function so it can be unit tested
    without needing a loaded model or any AWS/W&B dependency.
    """
    return pd.DataFrame([{
        "trip_distance": trip.trip_distance,
        "passenger_count": trip.passenger_count,
        "PULocationID": trip.PULocationID,
        "DOLocationID": trip.DOLocationID,
        "RatecodeID": trip.RatecodeID
    }])


def to_decimal(value) -> Decimal:
    """
    Convert a float to a DynamoDB-safe Decimal via string, to avoid
    binary floating-point artifacts (e.g. Decimal(0.1) picking up long
    trailing digits). DynamoDB rejects native Python floats outright.
    """
    return Decimal(str(value))


@app.get("/health")
def health_check():
    """
    Health Check Endpoint
    Verifies that the API server is running and the model is successfully loaded.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API is running, but the machine learning model failed to load."
        )
    return {"status": "ok", "message": "API is running and model is loaded"}


@app.post("/predict")
def predict(trip: TripInput):
    """
    Prediction Endpoint
    Takes trip attributes, formats them into a DataFrame, returns the predicted
    taxi fare, and logs the request/response to DynamoDB for monitoring.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Cannot make predictions."
        )

    start_time = time.perf_counter()

    # Convert incoming request body into a pandas DataFrame matching the feature layout
    input_data = build_input_dataframe(trip)

    # Generate prediction using the loaded pipeline
    prediction = float(model.predict(input_data)[0])
    predicted_fare = round(prediction, 2)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    prediction_id = str(uuid.uuid4())

    # Log the prediction to DynamoDB. A logging failure should never break
    # the prediction response itself, so this is wrapped separately.
    if prediction_table is not None:
        try:
            prediction_table.put_item(Item={
                "prediction_id": prediction_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trip_distance": to_decimal(trip.trip_distance),
                "passenger_count": trip.passenger_count,
                "PULocationID": trip.PULocationID,
                "DOLocationID": trip.DOLocationID,
                "RatecodeID": trip.RatecodeID,
                "predicted_fare": to_decimal(predicted_fare),
                "latency_ms": to_decimal(latency_ms),
                "model_version": MODEL_VERSION,
            })
        except Exception as e:
            print(f"Warning: failed to log prediction to DynamoDB: {e}")
    else:
        print("Warning: DynamoDB table not available, skipping prediction log.")

    return {
        "prediction_id": prediction_id,
        "predicted_fare_amount": predicted_fare
    }


@app.post("/feedback")
def submit_feedback(feedback: FeedbackInput):
    """
    Feedback Endpoint
    Records the actual observed fare for a previous prediction, so live
    accuracy can be calculated on the monitoring dashboard.
    """
    if prediction_table is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DynamoDB is not available. Cannot record feedback."
        )

    try:
        prediction_table.update_item(
            Key={"prediction_id": feedback.prediction_id},
            UpdateExpression="SET feedback_fare = :f",
            ExpressionAttributeValues={":f": to_decimal(feedback.actual_fare)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {e}"
        )

    return {"status": "feedback recorded", "prediction_id": feedback.prediction_id}
