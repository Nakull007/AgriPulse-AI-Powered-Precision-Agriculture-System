# =============================================================================
# APP.PY - Smart Sprinkler System Simulator (CORRECTED)
# =============================================================================

import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Agri-AI Sprinkler Simulator",
    page_icon="💧",
    layout="wide"
)

# --- Load Model and Assets (Cached for performance) ---
@st.cache_resource
def load_model():
    """Load the trained AI model from disk."""
    try:
        model = joblib.load('sprinkler_model.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# This dictionary is the "brain" for interpreting the model's output
ACTION_MAP = {
    0: {"icon": "✅", "message": "System OK. Conditions are optimal.", "color": "green"},
    1: {"icon": "💧", "message": "Watering... Soil is dry.", "color": "blue"},
    2: {"icon": "🚨", "message": "CRITICAL ALERT: High soil salinity detected! Watering to dilute.", "color": "red"},
    3: {"icon": "⚠️", "message": "WARNING: Low nutrients detected. Watering now. Schedule fertilization.", "color": "orange"},
    4: {"icon": "🌱", "message": "ALERT: Soil is wet but lacks nutrients. Fertigation is recommended.", "color": "orange"}
}

# =============================================================================
#  NEW FUNCTION TO PREPROCESS LIVE DATA
# =============================================================================
def preprocess_live_data(live_df):
    """
    Takes the live data from the UI and transforms it to match the format
    the model was trained on.
    """
    # Define the exact order of columns the model expects
    # This should match the order in your X_train.csv
    expected_columns = [
        'season_Monsoon', 'season_Post-Monsoon', 'season_Pre-Monsoon', 'season_Winter',
        'soil_moisture', 'temperature', 'humidity', 'rain_probability',
        'time_of_day', 'soil_ec'
    ]
    
    # Start with a DataFrame of zeros with the correct columns and order
    processed_df = pd.DataFrame(0, index=live_df.index, columns=expected_columns)
    
    # One-hot encode the 'season' column
    # For each row, set the correct season column to 1
    for i, row in live_df.iterrows():
        season_col_name = f"season_{row['season']}"
        if season_col_name in processed_df.columns:
            processed_df.at[i, season_col_name] = 1
            
    # Copy over the numerical features
    for col in live_df.columns:
        if col != 'season' and col in processed_df.columns:
            processed_df[col] = live_df[col]
            
    return processed_df


# --- Main Application UI ---

st.title("💧 Agri-AI Sprinkler System Simulator")
st.markdown("This app demonstrates our AI model's decision-making process in real-time. Adjust the sliders on the left to simulate different environmental conditions and see the AI's response.")

if model is None:
    st.error("Model file ('sprinkler_model.pkl') not found. Please make sure it's in the same directory as this app.")
else:
    # --- Sidebar for User Inputs ---
    st.sidebar.header("Simulate Sensor Inputs")

    soil_moisture = st.sidebar.slider(
        "Soil Moisture (Raw ADC)", 250, 950, 500,
        help="Lower values mean wet soil, higher values mean dry soil."
    )
    
    soil_ec = st.sidebar.slider(
        "Soil EC (mS/cm)", 0.5, 4.5, 1.8, step=0.1,
        help="Electrical Conductivity, a proxy for nutrient/mineral levels."
    )

    temperature = st.sidebar.slider("Temperature (°C)", 5, 45, 25)
    
    humidity = st.sidebar.slider("Humidity (%)", 20, 100, 75)
    
    rain_probability = st.sidebar.slider("Rain Probability", 0.0, 1.0, 0.1, step=0.05)
    
    time_of_day = st.sidebar.slider("Time of Day (24-hour)", 0, 23, 14)
    
    season = st.sidebar.selectbox(
        "Season",
        ('Monsoon', 'Post-Monsoon', 'Winter', 'Pre-Monsoon')
    )

    # --- Prediction and Output Display ---

    # 1. Create a DataFrame with the raw user input
    live_data_raw = pd.DataFrame([{
        'season': season,
        'soil_moisture': soil_moisture,
        'temperature': temperature,
        'humidity': humidity,
        'rain_probability': rain_probability,
        'time_of_day': time_of_day,
        'soil_ec': soil_ec
    }])
    
    # 2. Preprocess this raw data to match the model's training format
    live_data_processed = preprocess_live_data(live_data_raw)

    # 3. Make a prediction using the PROCESSED data
    predicted_action_code = model.predict(live_data_processed)[0]
    
    # 4. Get the corresponding description
    action_details = ACTION_MAP[predicted_action_code]

    # --- Display Results in the Main Panel ---
    st.header("AI Decision Output")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Soil Moisture", soil_moisture)
    col2.metric("Soil EC", f"{soil_ec} mS/cm")
    col3.metric("Temperature", f"{temperature} °C")
    
    st.divider()

    if action_details["color"] == "green":
        st.success(f"{action_details['icon']} **Decision:** {action_details['message']}", icon="✅")
    elif action_details["color"] == "blue":
        st.info(f"{action_details['icon']} **Decision:** {action_details['message']}", icon="💧")
    elif action_details["color"] == "orange":
        st.warning(f"{action_details['icon']} **Decision:** {action_details['message']}", icon="⚠️")
    elif action_details["color"] == "red":
        st.error(f"{action_details['icon']} **Decision:** {action_details['message']}", icon="🚨")

    with st.expander("See How the AI Made This Decision"):
        st.write("""
        Our trained RandomForest model analyzed the inputs you provided. Based on patterns learned from thousands of scenarios specific to North East Indian tea gardens, it determined the optimal action.
        """)
        st.subheader("Data Sent to Model (After Preprocessing)")
        st.dataframe(live_data_processed)
        st.subheader("Raw Prediction Output")
        st.json({
            "predicted_action_code": predicted_action_code,
            "details": action_details
        })