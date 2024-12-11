# Weather Data Prediction System

## Overview
This project is a weather data prediction system designed for renewable energy operations. It utilizes machine learning models to predict solar radiation and related variables based on meteorological data. The app also includes a user-friendly interface built with Streamlit for interactive data visualization and prediction.

## Features
- Real-time integration with OpenWeatherMap API for weather data.
- Predicts solar radiation and other metrics using advanced machine learning models.
- Provides interactive visualization of prediction results.
- Compares performance between multiple models (Random Forest and Gradient Boosting).

## Requirements
Ensure you have the following installed:
- Python 3.7+
- Required Python libraries (see `requirements.txt`)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/weather-prediction
   cd weather-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenWeatherMap API key by replacing `"your_api_key_here"` in the script with your API key.

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run weather_prediction_app.py
   ```

2. Open the URL provided by Streamlit in your browser (usually `http://localhost:8501`).

3. Interact with the application:
   - Select variables for prediction.
   - View interactive visualizations of prediction results.

## Project Structure
- `weather_prediction_app.py`: Main application script.
- `requirements.txt`: List of required Python libraries.
- `README.md`: Documentation for the project.

## Future Enhancements
- Add additional weather data sources for improved accuracy.
- Integrate time-series forecasting models like LSTMs.
- Automate daily data fetching and prediction scheduling.
