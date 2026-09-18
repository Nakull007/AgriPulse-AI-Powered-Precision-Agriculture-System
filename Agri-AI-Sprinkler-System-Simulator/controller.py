import time
import joblib # To load the trained model
import random # To simulate sensor readings for this example

# --- Action Interpretation and Alerting System ---
# This dictionary maps the model's output (0-4) to concrete actions and messages.

ACTION_MAP = {
    0: {"sprinkle": "OFF", "message": "System OK. Conditions are optimal."},
    1: {"sprinkle": "ON",  "message": "Watering... Soil is dry."},
    2: {"sprinkle": "ON",  "message": "CRITICAL ALERT: High soil salinity! Watering to dilute. Manual check required."},
    3: {"sprinkle": "ON",  "message": "WARNING: Low nutrients. Watering now. Schedule fertilization."},
    4: {"sprinkle": "OFF", "message": "ALERT: Soil is wet but lacks nutrients. Fertigation recommended."}
}

# --- Placeholder Functions for Hardware and Alerts ---

def get_sensor_data():
    """
    In a real system, this reads from your sensors (MCP3008, DHT22, etc.).
    Here, we simulate it to test the logic.
    """
    # Simulate data that might trigger different alerts
    simulated_data = {
        'soil_moisture': random.randint(250, 950),
        'temperature': random.uniform(18, 30),
        'humidity': random.randint(65, 95),
        'rain_probability': random.uniform(0.0, 1.0),
        'time_of_day': time.localtime().tm_hour,
        'soil_ec': random.uniform(0.8, 4.0)
    }
    print(f"\n[Sensor Read] Moisture: {simulated_data['soil_moisture']}, EC: {simulated_data['soil_ec']:.2f}, Rain: {simulated_data['rain_probability']:.2f}")
    return simulated_data

def control_sprinkler(command):
    """Controls the GPIO pin connected to the sprinkler relay."""
    if command == "ON":
        print("[Hardware Action] Turning sprinkler relay ON.")
        # GPIO.output(RELAY_PIN, GPIO.HIGH)
    else:
        print("[Hardware Action] Turning sprinkler relay OFF.")
        # GPIO.output(RELAY_PIN, GPIO.LOW)

def send_alert_to_device(message, priority="NORMAL"):
    """
    This function sends the alert. This is where 5G is key.
    It can be expanded to use SMS, Email, or a Dashboard.
    """
    print(f"[Alert System] Priority: {priority} | Message: {message}")
    if priority in ["HIGH", "CRITICAL"]:
        # --- Integrate your alert service here ---
        # e.g., send_sms_via_twilio(message)
        # e.g., publish_to_iot_dashboard(message, priority)
        print("[Alert System] High-priority alert dispatched via 5G network.")

# --- Main Application Logic ---

def main():
    # Load your trained machine learning model
    # You would train this model using the CSV file we generated.
    try:
        model = joblib.load('sprinkler_model.pkl') # Make sure you have a trained model file
        print("AI model loaded successfully.")
    except FileNotFoundError:
        print("Error: 'sprinkler_model.pkl' not found. Please train the model first.")
        return

    while True:
        # 1. Gather all inputs
        current_data = get_sensor_data()
        
        # 2. Format data for the model
        # The order must be EXACTLY the same as during training!
        features = [
            current_data['soil_moisture'],
            current_data['temperature'],
            current_data['humidity'],
            current_data['rain_probability'],
            current_data['time_of_day'],
            current_data['soil_ec']
        ]
        
        # 3. Get a decision from the AI model
        predicted_action_code = model.predict([features])[0] # model.predict returns a list, e.g., [2]
        
        # 4. Interpret and execute the decision
        if predicted_action_code in ACTION_MAP:
            action = ACTION_MAP[predicted_action_code]
            
            # Execute sprinkler control
            control_sprinkler(action["sprinkle"])
            
            # Send status message and alerts
            alert_message = action["message"]
            if "ALERT" in alert_message or "WARNING" in alert_message:
                send_alert_to_device(alert_message, priority="HIGH")
            else:
                send_alert_to_device(alert_message, priority="NORMAL")
        else:
            print(f"Error: Model predicted an unknown action code: {predicted_action_code}")

        # 5. Wait for the next cycle
        print("--- Waiting for 30 seconds before next check ---")
        time.sleep(30)

if __name__ == "__main__":
    # To run this example, you first need to create a dummy model file.
    # You can do this after training your real model. For now, you could
    # create a placeholder if you just want to run the script.
    main()