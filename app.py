
# Import the libraries required by the application
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configure the Streamlit page
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# Load the saved Random Forest model
MODEL_PATH = "crop_recommendation_model.joblib"
model = joblib.load(MODEL_PATH)

# Display the application heading and introduction
st.title("🌱 Crop Recommendation System")

st.write(
    "Enter your soil nutrient and weather measurements "
    "to receive a crop recommendation from our trained "
    "Random Forest machine learning model."
)

st.info(
    "For the most meaningful result, use measurements "
    "obtained from reliable soil testing and weather sources."
)

# Create the input form
st.subheader("Enter your soil and weather measurements")

with st.form("crop_recommendation_form"):

    # Soil nutrient inputs
    nitrogen = st.number_input(
        "Nitrogen (N) — Range: 0–140",
        min_value=0.0,
        max_value=140.0,
        value=23.0,
        step=1.0
    )

    phosphorus = st.number_input(
        "Phosphorus (P) — Range: 5–145",
        min_value=5.0,
        max_value=145.0,
        value=67.0,
        step=1.0
    )

    potassium = st.number_input(
        "Potassium (K) — Range: 5–205",
        min_value=5.0,
        max_value=205.0,
        value=45.0,
        step=1.0
    )

    # Environmental inputs
    temperature = st.number_input(
        "Temperature (°C) — Range: 8–44",
        min_value=8.0,
        max_value=44.0,
        value=34.0,
        step=0.1
    )

    humidity = st.number_input(
        "Humidity (%) — Range: 14–100",
        min_value=14.0,
        max_value=100.0,
        value=35.0,
        step=0.1
    )

    soil_ph = st.number_input(
        "Soil pH — Range: 3.5–10",
        min_value=3.5,
        max_value=10.0,
        value=4.6,
        step=0.1
    )

    rainfall = st.number_input(
        "Rainfall (mm) — Range: 20–300",
        min_value=20.0,
        max_value=300.0,
        value=70.0,
        step=1.0
    )

    # Submit button for the prediction form
    submitted = st.form_submit_button(
        "Recommend Crop",
        use_container_width=True
    )

# Generate a prediction after the user submits the form
if submitted:

    # Arrange the inputs in the same order used during model training
    input_data = pd.DataFrame(
        [[
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            soil_ph,
            rainfall
        ]],
        columns=[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]
    )

    # Predict the most suitable crop according to the trained model
    prediction = model.predict(input_data)[0]

    # Obtain estimated probabilities for all crop classes
    probabilities = model.predict_proba(input_data)[0]

    # Match each probability to its corresponding crop label
    crop_probabilities = pd.DataFrame({
        "Crop": model.classes_,
        "Probability": probabilities
    })

    # Sort the crops by probability, from highest to lowest
    top_three = crop_probabilities.sort_values(
        by="Probability",
        ascending=False
    ).head(3)

    # Display the primary recommendation
    st.success(f"Recommended crop: **{str(prediction).upper()}**")

    # Display the three highest crop probability estimates
    st.subheader("Top 3 Crop Recommendations")

    for _, row in top_three.iterrows():
        crop_name = str(row["Crop"])
        probability = float(row["Probability"])

        st.write(f"**{crop_name.upper()} — {probability:.1%}**")
        st.progress(probability)

    # Explain what the probability scores mean
    st.caption(
        "These percentages are model-estimated class probabilities. "
        "They do not guarantee crop suitability, yield, or profitability."
    )

    # Display the values used to generate the prediction
    with st.expander("View submitted measurements"):
        st.dataframe(input_data, use_container_width=True)
