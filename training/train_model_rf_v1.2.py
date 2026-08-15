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


# Initialize Weights & Biases tracking
run = wandb.init(
    entity="models-university-of-denver9526",
    project="DU_Summer25_Final_Project_Taxi_Fare",
    name="random-forest-v2.1-tuned",
    config={
        "architecture": "RandomForestRegressor_v2.1_GridSearch",
        "dataset_source": (
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-04.parquet"
        ),
        "sample_size": 50000,
        "test_size": 0.2,
        "n_iter": 10,
        "git_commit": get_git_commit(),
    },
)

# Reads in the dataset
print("Loading dataset...")
url = wandb.config.dataset_source
df = pd.read_parquet(url)

print("Sampling data...")
df = df.sample(n=wandb.config.sample_size, random_state=42)

# Drop N/A values from all rows
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

# Sets features
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

# Set feature types
numeric_features = ["trip_distance", "passenger_count"]
categorical_features = ["PULocationID", "DOLocationID", "RatecodeID"]

# ColumnTransormer for scaling and encoding
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

# --- Version 2.1 Expanded/Alternative Hyperparameter Grid ---
param_grid = {
    "model__n_estimators": [150, 250, 350],
    "model__max_depth": [None, 15, 25],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.8, 1.0],
}

grid_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)

print("Training and tuning Random Forest v2.1 model...")
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Update W&B config with the optimized hyperparameters found via search
wandb.config.update(best_params)

best_model = grid_search.best_estimator_

# --- Evaluation Metrics Section ---
print("Evaluating model on test data...")
y_pred = best_model.predict(X_test)

# Model performance metrics
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

print("--- Test Set Performance ---")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Error (MAE):     ${mae:.2f}")
print(f"R-squared (R²):                {r2:.4f}")

# Log evaluation metrics to W&B
run.log({"test_rmse": rmse, "test_mae": mae, "test_r2": r2})

# Save model
model_filename = "taxi_fare_random_forest.pkl"
joblib.dump(best_model, model_filename)
print(
    "Random Forest v2.1 model trained and saved successfully as"
    f" '{model_filename}'!"
)

# --- W&B Artifact Versioning ---
print("Logging model artifact to W&B...")
artifact = wandb.Artifact(
    name="taxi-fare-random-forest",
    type="model",
    description=(
        "Expanded grid-searched Random Forest v2.1 trained on NYC yellow taxi"
        " dataset."
    ),
    metadata=dict(wandb.config),
)
artifact.add_file(model_filename)

# Log the artifact directly to the run (versioned model tracking)
run.log_artifact(artifact)

# Finish the W&B run
run.finish()
print("Pipeline complete and artifact logged successfully!")
