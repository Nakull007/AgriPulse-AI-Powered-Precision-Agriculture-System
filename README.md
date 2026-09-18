# 💧 AgriPulse: AI-Powered Precision Agriculture System

This project presents a "Digital Twin" of an AI-powered smart sprinkler system designed for precision agriculture, specifically tailored for the unique needs of a North East Indian tea garden. Instead of just watering on a schedule, this system uses an AI model to make intelligent, real-time decisions based on a variety of environmental factors.

➡️ **[Try the Live App Here!](#)** ⬅️
*(Note: Replace the `#` with your actual live app URL)*

---

## 🚀 The Problem

Traditional irrigation systems are inefficient. They waste water, are oblivious to weather forecasts, and fail to account for the specific needs of different crops or the nutrient levels in the soil. This leads to resource wastage, poor crop health, and reduced yield.

## 💡 Our Solution: An AI-Powered "Digital Agronomist"

Our AgriPulse system solves this by acting as an intelligent agent. It gathers data from multiple sources, processes it through a trained machine learning model, and makes a holistic decision.

### Key Features:

* **Multi-Factor Decision Making:** The system doesn't just check soil moisture. It analyzes:
  * Soil Moisture
  * Soil EC (Electrical Conductivity as a proxy for nutrient levels)
  * Ambient Temperature & Humidity
  * Real-time Rain Probability from weather APIs
  * Time of Day & Geo-specific Season
* **Intelligent Alerts:** The system goes beyond a simple ON/OFF. It can issue specific alerts, such as:
  * *High Salinity Warning:* Waters to dilute soil but alerts the farmer to a toxic condition.
  * *Fertigation Alert:* Recommends fertilization when soil is wet but nutrients are low.
* **5G-Enabled Vision:** The project is built with 5G in mind, enabling massive IoT deployments on smart farms, real-time data streaming to cloud dashboards, and instant alerts to farmers anywhere in the world.

## 🤖 The AI Model

The core of our system is a `RandomForestClassifier` model trained on a custom, high-fidelity synthetic dataset.

* **Hyper-Local Data:** We didn't use generic data. We generated 4,000+ data points that simulate the specific subtropical monsoon climate of North East India, capturing its unique seasonal patterns.
* **Expert-Driven Logic:** The dataset was labeled using a complex set of rules that mimic the decisions of an expert agronomist, creating a robust "answer key" for the model to learn from.
* **High Performance:** The final model achieved ~99.9% accuracy on unseen test data, with near-perfect precision and recall across all five decision states.

*(Action: Replace this text with a screenshot of your actual Confusion Matrix plot)*

## 🛠️ How to Use the Simulator

Our live web application allows you to interact with the AI model directly, without any hardware.

1. Navigate to the **[Live App](#)**.
2. Use the sliders and dropdowns on the left sidebar to simulate different sensor readings and environmental conditions.
3. Observe the **AI Decision Output** panel on the right, which updates in real-time to show the model's decision and a descriptive message.

*(Action: Replace this text with a screenshot of your Streamlit app in action)*

## ⚙️ Running the Project Locally

If you wish to run the simulator on your own machine:

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/AgriPulse.git](https://github.com/YourUsername/AgriPulse.git)
cd AgriPulse
