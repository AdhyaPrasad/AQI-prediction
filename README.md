## National Air Quality Prediction System
A machine learning-based web application to forecast Air Quality Index (AQI) for Indian states using historical pollutant data. Built with XGBoost and visualized using Streamlit, this system helps users monitor future AQI trends, understand monthly health risks, and plan accordingly.

### Features:
State-wise AQI Prediction for future years
Monthly AQI Summary with health risk categorization
Time Series Visualization of predicted AQI
Health Risk Gauge and personalized advisory
Precomputed forecasts stored via SQLite for faster loading

**Data Source:**
Air quality data used in this project is from https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023

##### *Disclaimer*
This system provides predicted AQI values using machine learning models trained on historical data.
For official and real-time AQI, refer to CPCB – National AQI.
States like Arunachal Pradesh and Jharkhand are excluded due to insufficient data.

