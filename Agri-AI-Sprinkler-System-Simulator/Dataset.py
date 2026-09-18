import csv
import random

# --- Geo-Specific Climate Scenarios for North East India ---
# We define distinct weather patterns to simulate the region's climate.

def get_seasonal_scenario():
    """Randomly returns a climate scenario based on North East India's seasons."""
    seasons = {
        'monsoon': 0.45,       # Longest season (June-Sept)
        'post_monsoon': 0.20, # Cool and clearer (Oct-Nov)
        'winter': 0.15,        # Cool, foggy, and dry (Dec-Feb)
        'pre_monsoon': 0.20  # Hotter, with thunderstorms (Mar-May)
    }
    
    # Choose a season based on its weighted probability
    chosen_season = random.choices(list(seasons.keys()), weights=list(seasons.values()), k=1)[0]
    
    if chosen_season == 'monsoon':
        return {
            'name': 'Monsoon',
            'temp_range': (22, 30),
            'humidity_range': (85, 98),
            'rain_prob_range': (0.5, 1.0) # Frequent, heavy rain
        }
    elif chosen_season == 'post_monsoon':
        return {
            'name': 'Post-Monsoon',
            'temp_range': (18, 28),
            'humidity_range': (70, 85),
            'rain_prob_range': (0.1, 0.4) # Less frequent rain
        }
    elif chosen_season == 'winter':
        return {
            'name': 'Winter',
            'temp_range': (12, 22), # Cooler temperatures
            'humidity_range': (65, 80),
            'rain_prob_range': (0.0, 0.15) # Very little rain
        }
    else: # pre_monsoon
        return {
            'name': 'Pre-Monsoon',
            'temp_range': (25, 33), # Getting hotter
            'humidity_range': (60, 85),
            'rain_prob_range': (0.1, 0.6) # Occasional heavy thunderstorms
        }

# --- Agronomical Thresholds for Tea (from previous script) ---
MOISTURE_WET_THRESHOLD = 450
MOISTURE_DRY_THRESHOLD = 550
EC_LOW_THRESHOLD = 1.2
EC_HIGH_THRESHOLD = 2.8
RAIN_PROBABILITY_THRESHOLD = 0.35

# --- Unified Action State Definitions ---
ACTION_STATES = {
    "DO_NOTHING": 0, "SPRINKLE_NORMAL": 1, "SPRINKLE_AND_ALERT_HIGH_EC": 2,
    "SPRINKLE_AND_WARN_LOW_EC": 3, "ALERT_FERTIGATE": 4
}

# --- Dataset Generation Settings ---
NUM_ROWS = 4000
FILENAME = "NE_India_tea_garden_data.csv"

def generate_row():
    """Generates a single data row based on a randomly chosen seasonal scenario."""
    
    # 1. Get a realistic climate context for this specific data point
    scenario = get_seasonal_scenario()
    
    # 2. Generate environmental data based on this specific climate scenario
    temperature = round(random.uniform(*scenario['temp_range']), 1)
    humidity = random.randint(*scenario['humidity_range'])
    rain_probability = round(random.uniform(*scenario['rain_prob_range']), 2)
    time_of_day = random.randint(0, 23)
    
    # 3. Generate sensor values
    soil_moisture = random.randint(250, 950)
    # Reflecting soil leaching: EC is more likely to be on the lower side.
    soil_ec = round(random.triangular(0.7, 3.5, 1.5), 2) # (min, max, mode) - skews towards lower EC

    # 4. Apply the expert decision logic
    is_dry = soil_moisture > MOISTURE_DRY_THRESHOLD
    is_wet = soil_moisture < MOISTURE_WET_THRESHOLD
    ec_is_low = soil_ec < EC_LOW_THRESHOLD
    ec_is_high = soil_ec > EC_HIGH_THRESHOLD
    ec_is_normal = not ec_is_low and not ec_is_high

    action_state = ACTION_STATES["DO_NOTHING"]

    if rain_probability > RAIN_PROBABILITY_THRESHOLD:
        action_state = ACTION_STATES["DO_NOTHING"]
    elif is_dry:
        if ec_is_normal: action_state = ACTION_STATES["SPRINKLE_NORMAL"]
        elif ec_is_low: action_state = ACTION_STATES["SPRINKLE_AND_WARN_LOW_EC"]
        else: action_state = ACTION_STATES["SPRINKLE_AND_ALERT_HIGH_EC"]
    elif is_wet and ec_is_low:
        action_state = ACTION_STATES["ALERT_FERTIGATE"]
            
    return {
        'season': scenario['name'], # Adding season for context!
        'soil_moisture': soil_moisture, 'temperature': temperature, 'humidity': humidity,
        'rain_probability': rain_probability, 'time_of_day': time_of_day,
        'soil_ec': soil_ec, 'action_state': action_state
    }

def main():
    header = [
        'season', 'soil_moisture', 'temperature', 'humidity', 'rain_probability', 
        'time_of_day', 'soil_ec', 'action_state'
    ]
    with open(FILENAME, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for _ in range(NUM_ROWS):
            writer.writerow(generate_row())
            
    print(f"Successfully created a hyper-local North East India Tea Garden dataset: '{FILENAME}'!")

if __name__ == "__main__":
    main()