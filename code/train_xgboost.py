import os
import pandas as pd
import xgboost as xgb
from aqi_functions import create_features, create_lag_features, impute_missing_lags

# Configuration
STATES = ["Andhra_Pradesh", "Karnataka"]  # Test with 2 states first
MODEL_DIR = "models/state_models"
DATA_DIR = "data/processed_states"
REQUIRED_COLS = ['PM2.5 (ug/m3)', 'hour', 'month', 'pm_lag_1Y']

def train_state_model(state):
    try:
        # 1. Load data
        file_path = f"{DATA_DIR}/{state}_processed.csv"
        df = pd.read_csv(
            file_path,
            parse_dates=['datetime'],
            index_col='datetime',
            dtype={'PM2.5 (ug/m3)': 'float32'},
            low_memory=False
        )
        
        # 2. Feature engineering
        df = create_features(df)  # Creates hour/month/etc
        df = create_lag_features(df)
        df = impute_missing_lags(df, 'PM2.5 (ug/m3)')
        
        # 3. Validate FINAL features
        missing = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
            
        # 4. Train model
        train_size = int(len(df) * 0.8)
        X_train = df[REQUIRED_COLS].iloc[:train_size]
        y_train = df['PM2.5 (ug/m3)'].iloc[:train_size]
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        model.fit(X_train, y_train)
        
        # 5. Save
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_model(f"{MODEL_DIR}/xgboost_{state}_model.json")
        print(f"✅ Trained {state}")
        
    except Exception as e:
        print(f"❌ Failed {state}: {str(e)}")

if __name__ == "__main__":
    for state in STATES:
        train_state_model(state)
