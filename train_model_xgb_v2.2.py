# Load libraries
import subprocess
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import wandb
from xgboost import XGBRegressor


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


# Sets up the Weights & Biases tracking
run = wandb.init(
    entity="models-university-of-denver9526",
    project="DU_Summer25_Final_Project_Taxi_Fare",
    name="taxi-fare-xgboost-tuned",
    config={
        "architecture": "XGBRegressor_ExpandedGridSearch_LargeSample",
        "dataset_source": (
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-04.parquet"
        ),
        "sample_size": 500000,
        "test_size": 0.2,
        "n_iter": 5,
        "git_commit": get_git_commit(),
    },
)

# Read in the data set
url = wandb.config.dataset_source
df = pd.read_parquet(url)

# Create subset of data due to size and causing WSL to crash
df = df.sample(n=wandb.config.sample_size, random_state=42)

# Clean data
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

# Sets up features to be used in the model
features = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]
X = df[features]
y = df["fare_amount"]

# Sets up train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=wandb.config.test_size, random_state=42
)

# Sets up numeric and categorical features
numeric_features = ["trip_distance", "passenger_count"]
categorical_features = ["PULocationID", "DOLocationID", "RatecodeID"]

# Sets up scalers for numeric and categorical features
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

# Sets up pipeline for xgboost regressor
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(random_state=42, n_jobs=-1)),
    ]
)

# Paramaters for search
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "model__learning_rate": [0.01, 0.1, 0.2],
}

# Runs RandomizedSearchCV funtion
grid_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)

# Fits to train data
grid_search.fit(X_train, y_train)

# Selects the best params
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Update W&B with optimized hyperparameters
wandb.config.update(best_params)

best_model = grid_search.best_estimator_

# Evaluating the metrics with test set
y_pred = best_model.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Error (MAE):     ${mae:.2f}")
print(f"R-squared (R²):                {r2:.4f}")

# Log metrics
run.log({"test_rmse": rmse, "test_mae": mae, "test_r2": r2})

# Save model with name taxi_fare_xgb_tuned.pkl
model_filename = "taxi_fare_xgb_tuned.pkl"
joblib.dump(best_model, model_filename)
print(
    "XGBoost model trained and saved successfully as"
    f" '{model_filename}'!"
)

# WandB articfact logging
print("Logging model artifact to W&B...")
artifact = wandb.Artifact(
    name="taxi-fare-xgboost-tuned",
    type="model",
    description=(
        "Tuned XGBoost Regressor trained on 500k sample NYC yellow taxi"
        " dataset."
    ),
    metadata=dict(wandb.config),
)
artifact.add_file(model_filename)

run.log_artifact(artifact)

# finishing the run
run.finish()
print("Pipeline complete and artifact logged successfully!")