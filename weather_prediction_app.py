import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import requests
import streamlit as st

# 1. Data Loading (Real-time API Integration - OpenWeatherMap as example)
def fetch_weather_data(api_key, city="London"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_data = {
            'wind_speed': data['wind']['speed'],
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'solar_radiation': np.random.uniform(0, 1000)  # Placeholder as API may not provide
        }
        return weather_data
    else:
        st.error("Failed to fetch data from API")
        return None

# Placeholder for API Key
API_KEY = "your_api_key_here"

# Example usage of API
data = []
for _ in range(1000):  # Simulate 1000 data points
    city_data = fetch_weather_data(API_KEY, city="London")
    if city_data:
        data.append(city_data)

# Convert to DataFrame if data is available
if data:
    data_df = pd.DataFrame(data)
else:
    st.error("No data fetched from the API. Please check your API key and internet connection.")
    st.stop()

# 2. Data Preprocessing
# Feature Engineering (Example: Calculating Air Density)
data_df['air_density'] = data_df['pressure'] / (287.05 * (data_df['temperature'] + 273.15))
data_df['wind_power'] = 0.5 * data_df['air_density'] * (data_df['wind_speed'] ** 3)

# Define Features and Target Variables
features = data_df[['wind_speed', 'temperature', 'humidity', 'pressure', 'air_density', 'wind_power']]
target = data_df['solar_radiation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 3. Model Training
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

model_results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    model_results[model_name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# Select the best model
best_model_name = max(model_results, key=lambda x: model_results[x]['R2'])
best_model = models[best_model_name]

st.title("Weather Data Prediction")

st.write("## Model Evaluation Results")
for model_name, metrics in model_results.items():
    st.write(f"### {model_name}")
    for metric_name, value in metrics.items():
        st.write(f"- {metric_name}: {value:.4f}")

if st.button("Predict with Best Model"):
    predictions = best_model.predict(X_test)
    st.write(f"Using Best Model: {best_model_name}")
    st.write(f"R2 Score: {model_results[best_model_name]['R2']:.4f}")

    # Streamlit-native visualization
    comparison_df = pd.DataFrame({"True Values": y_test.values, "Predictions": predictions})
    st.line_chart(comparison_df)
