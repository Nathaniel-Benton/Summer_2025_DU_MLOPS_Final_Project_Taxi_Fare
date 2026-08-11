import subprocess
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import wandb


# Helper function to track Git commit hash
def get_git_commit():
  try:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .strip()
        .decode("utf-8")
    )
  except Exception:
    return "not_a_git_repo"


# 1. Initialize Weights & Biases tracking
run = wandb.init(
    entity="models-university-of-denver9526",
    project="DU_Summer25_Final_Project_Taxi_Fare",
    config={
        "architecture": "RandomForestRegressor",
        "dataset_source": (
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-04.parquet"
        ),
        "sample_size": 50000,
        "test_size": 0.2,
        "n_estimators": 50,
        "git_commit": get_git_commit(),
    },
)

# Load your NYC Taxi DataFrame
print("Loading dataset...")
url = wandb.config.dataset_source
df = pd.read_parquet(url)

# Subset for initial training purposes
df = df.sample(n=wandb.config.sample_size, random_state=42)

# Quick data cleaning & feature selection
df = df.dropna(
    subset=[
        "total_amount",
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "RatecodeID",
    ]
)

# Split into features
features = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]
X = df[features]
y = df["fare_amount"]

# Set xtrain and ytrain
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=wandb.config.test_size, random_state=42
)

# Set numerical and categorical columns
numeric_features = ["trip_distance", "passenger_count"]
categorical_features = ["PULocationID", "DOLocationID", "RatecodeID"]

# ColumnTransformer setup
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

# Random Forest Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=wandb.config.n_estimators, random_state=42
            ),
        ),
    ]
)

# Fit the model directly
print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate model performance
print("Evaluating model...")
y_pred = pipeline.predict(X_test)
rmse = float((mean_squared_error(y_test, y_pred)) ** 0.5)
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

print(f"Test RMSE: ${rmse:.2f}")
print(f"Test MAE:  ${mae:.2f}")
print(f"Test R²:   {r2:.4f}")

# Log metrics to W&B
run.log({"test_rmse": rmse, "test_mae": mae, "test_r2": r2})

# Save the model locally
model_filename = "taxi_fare_random_forest.pkl"
joblib.dump(pipeline, model_filename)

# --- W&B Model Registry & Artifact Versioning ---
print("Logging model as a W&B Artifact to the Model Registry...")
artifact = wandb.Artifact(
    name="taxi-fare-random-forest",
    type="model",
    description=(
        "Random Forest Regressor trained on NYC yellow taxi dataset for fare"
        " prediction."
    ),
    metadata=dict(wandb.config),
)
artifact.add_file(model_filename)

# Log the artifact and link it to the W&B Model Registry
run.log_artifact(artifact, aliases=["staging", "latest"])

# Finish the W&B run
run.finish()
print("Pipeline complete and artifact logged successfully!")