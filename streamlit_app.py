# Import libraries
import os

import requests
import streamlit as st

st.set_page_config(page_title="Taxi Fare Prediction")

#Backend location — defaults to local dev, but gets overridden via the
#API_URL env var in Docker
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

#  Sets the st.session_state
if "last_prediction_id" not in st.session_state:
    st.session_state.last_prediction_id = None
if "last_predicted_fare" not in st.session_state:
    st.session_state.last_predicted_fare = None

#Sets the title of the app
st.title("NYC Taxi Fare Prediction")
st.write("Enter trip details to get a predicted fare from the model.")

#batches all the inputs below so changing one widget doesn't trigger a rerun
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    #Sets input fields and populates them with values
    with col1:
        trip_distance = st.number_input(
            "Trip Distance (miles)", min_value=0.0, value=3.0, step=0.5
        )
        passenger_count = st.number_input(
            "Passenger Count", min_value=1, max_value=6, value=1, step=1
        )
        ratecode_id = st.selectbox(
            "Rate Code",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "1 - Standard rate",
                2: "2 - JFK",
                3: "3 - Newark",
                4: "4 - Nassau/Westchester",
                5: "5 - Negotiated fare",
                6: "6 - Group ride",
            }[x],
        )

# Binds the input so user can't submitting a zone ID outside the data
    with col2:
        pu_location_id = st.number_input(
            "Pickup Location ID", min_value=1, max_value=265, value=1, step=1
        )
        do_location_id = st.number_input(
            "Dropoff Location ID", min_value=1, max_value=265, value=2, step=1
        )

    #Creates predict fare button
    submitted = st.form_submit_button("Predict Fare")

#Requests post to the backend
if submitted:
    payload = {
        "trip_distance": trip_distance,
        "passenger_count": int(passenger_count),
        "PULocationID": int(pu_location_id),
        "DOLocationID": int(do_location_id),
        "RatecodeID": int(ratecode_id),
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        # Stash the prediction_id so the feedback form below can attach
        # to this specific prediction after the page reruns.
        st.session_state.last_prediction_id = result["prediction_id"]
        st.session_state.last_predicted_fare = result["predicted_fare_amount"]

        # Predicted fare result printed
        st.success(f"Predicted fare: ${result['predicted_fare_amount']:.2f}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not reach FASTAPI. Make sure the FastAPI backend is running."
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Only appears after a successful prediction this session, since feedback
# needs a real prediction_id to attach to.
if st.session_state.last_prediction_id:
    st.divider()
    st.subheader("Enter the actual fare of the trip for monitoring the model.")
    st.write(f"Predicted: ${st.session_state.last_predicted_fare:.2f}")

    with st.form("feedback_form"):
        actual_fare = st.number_input(
            "Actual Fare ($)", min_value=0.0, value=0.0, step=0.5
        )
        feedback_submitted = st.form_submit_button("Submit Actual Fare")

    if feedback_submitted:
        try:
            fb_response = requests.post(
                f"{API_URL}/feedback",
                json={
                    "prediction_id": st.session_state.last_prediction_id,
                    "actual_fare": actual_fare,
                },
                timeout=10,
            )
            fb_response.raise_for_status()
            st.success("Thanks! Feedback recorded.")
        except Exception as e:
            st.error(f"Failed to submit feedback: {e}")