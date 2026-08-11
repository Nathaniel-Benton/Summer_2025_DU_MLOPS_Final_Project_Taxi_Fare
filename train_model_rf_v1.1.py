import subprocess
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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
        "architecture": "RandomForestRegressor_GridSearch",
        "dataset_source": (
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-04.parquet"
        ),
        "sample_size": 50000,
        "test_size": 0.2,
        "git_commit": get_git_commit(),
    },
)

print("Loading dataset...")
url = wandb.config.dataset_source
df = pd.read_parquet(url)

print("Sampling data...")
df = df.sample(n=wandb.config.sample_size, random_state=42)

print("Cleaning data...")
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

features = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]
X = df[features]
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=wandb.config.test_size, random_state=42
)

numeric_features = ["trip_distance", "passenger_count"]
categorical_features = ["PULocationID", "DOLocationID", "RatecodeID"]

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
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ]
)

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [5, 10, 15, None],
    "model__min_samples_split": [2, 5, 10],
}

grid_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)

print("Training and tuning Random Forest model...")
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Update W&B config with the optimized hyperparameters found via search
wandb.config.update(best_params)

best_model = grid_search.best_estimator_

# --- Evaluation Metrics Section ---
print("Evaluating model on test data...")
y_pred = best_model.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

print("--- Test Set Performance ---")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Error (MAE):     ${mae:.2f}")
print(f"R-squared (R²):                {r2:.4f}")

# Log evaluation metrics to W&B
run.log({"test_rmse": rmse, "test_mae": mae, "test_r2": r2})

# Save model locally
model_filename = "taxi_fare_random_forest.pkl"
joblib.dump(best_model, model_filename)
print(f"Random Forest model trained and saved successfully as '{model_filename}'!")

# --- W&B Artifact Versioning ---
print("Logging model artifact to W&B...")
artifact = wandb.Artifact(
    name="taxi-fare-random-forest",
    type="model",
    description=(
        "Tuned Random Forest Regressor trained on NYC yellow taxi dataset."
    ),
    metadata=dict(wandb.config),
)
artifact.add_file(model_filename)

# Log the artifact directly to the run (this creates versioned model tracking)
run.log_artifact(artifact)

# Finish the W&B run
run.finish()
print("Pipeline complete and artifact logged successfully!")