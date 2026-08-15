# Import libraries
import os
from decimal import Decimal

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Set page config title
st.set_page_config(page_title="Taxi Fare Model — Monitoring Dashboard")

TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "taxi-fare-predictions")


def _to_float(x):
    """DynamoDB returns numeric fields as Decimal."""
    return float(x) if isinstance(x, Decimal) else x


@st.cache_data(ttl=30)
def load_logs() -> pd.DataFrame:
    """Scan the DynamoDB table and return all logged predictions as a DataFrame."""
    try:
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(TABLE_NAME)

        items = []
        response = table.scan()
        items.extend(response.get("Items", []))

        # DynamoDB scan is paginated; keep pulling until there's no more data
        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response.get("Items", []))

    except Exception as e:
        st.error(
            f"Could not read from DynamoDB table '{TABLE_NAME}': {e}\n\n"
            "Make sure your AWS Academy lab session is running and credentials "
            "are current."
        )
        return pd.DataFrame()

    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)

    numeric_cols = [
        "trip_distance", "passenger_count", "PULocationID", "DOLocationID",
        "RatecodeID", "predicted_fare", "latency_ms", "feedback_fare",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    return df

# Sets the title for the page
st.title("Taxi Fare Model — Monitoring Dashboard")

# Set refresh button
col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("Refresh"):
        st.cache_data.clear()

# Loads logs and warns if empty
df = load_logs()

if df.empty:
    st.warning(
        "No prediction logs found yet. Send some requests first, "
        "or check your AWS connection."
    )
    st.stop()

# Displays how many logs are loaded
st.write(f"Loaded {len(df)} logged predictions.")

# --- Feedback / live accuracy ---
has_feedback = "feedback_fare" in df.columns
labeled_df = df[df["feedback_fare"].notna()].copy() if has_feedback else pd.DataFrame()

st.subheader("Accuracy from user feedback")

if not labeled_df.empty:
    labeled_df["abs_error"] = (labeled_df["predicted_fare"] - labeled_df["feedback_fare"]).abs()
    labeled_df["pct_error"] = labeled_df["abs_error"] / labeled_df["feedback_fare"].replace(0, pd.NA) * 100

    mae = labeled_df["abs_error"].mean()
    mape = labeled_df["pct_error"].mean()
    within_3_dollars = (labeled_df["abs_error"] <= 3).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error", f"${mae:.2f}")
    col2.metric("Mean Absolute % Error", f"{mape:.1f}%")
    col3.metric("Within $3 of actual", f"{within_3_dollars:.1f}%")

    if mae > 4:
        st.error(f"Mean absolute error is ${mae:.2f}, above the $4 threshold. Investigate the model.")
else:
    st.info("No feedback collected yet — submit actual fares via the prediction app to populate this.")

st.divider()

# --- Latency over time ---
st.subheader("Prediction Latency Over Time")

if "latency_ms" in df.columns and "timestamp" in df.columns:
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["latency_ms"], marker="o", linestyle="-", markersize=3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Latency (ms)")
    fig.autofmt_xdate()
    st.pyplot(fig)
else:
    st.info("No latency data available yet.")

st.divider()

# --- Distribution of predicted fares (target drift) ---
st.subheader("Distribution of Predicted Fares")

if "predicted_fare" in df.columns:
    fig, ax = plt.subplots()
    ax.hist(df["predicted_fare"], bins=30, alpha=0.7)
    ax.set_xlabel("Predicted Fare ($)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.info("No prediction data available yet.")

st.divider()

# --- Input feature drift: trip distance over time ---
st.subheader("Input Drift: Trip Distance Requested Over Time")

if "trip_distance" in df.columns and "timestamp" in df.columns:
    fig, ax = plt.subplots()
    ax.scatter(df["timestamp"], df["trip_distance"], alpha=0.5, s=15)
    ax.set_xlabel("Time")
    ax.set_ylabel("Trip Distance (miles)")
    fig.autofmt_xdate()
    st.pyplot(fig)
else:
    st.info("No trip distance data available yet.")

st.divider()

# --- Raw log table ---
st.subheader("Raw Prediction Logs")
st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)