import os
import pandas as pd
import xgboost as xgb
import numpy as np
from aqi_functions import create_features, create_lag_features, impute_missing_lags

STATES_DIR = "data/processed_states"
MODELS_DIR = "models/state_models"

ALTERNATE_TARGETS = {
    'Karnataka': 'PM10 (ug/m3)',#actually not pm2.5 or pm 10
    'Kerala': 'Ozone (ug/m3)'
}

FALLBACK_STRATEGIES = {
    'PM2.5 (ug/m3)': [
        lambda df: df['PM10 (ug/m3)'] * 0.7,
        lambda df: (df['CO (mg/m3)'] * 5) + (df['NOx (ug/m3)'] * 0.2)
    ],
    'PM10 (ug/m3)': [
        lambda df: df['PM2.5 (ug/m3)'] * 1.4,
        lambda df: df['Ozone (ug/m3)'] * 2.0
    ]
}


def get_target(state, df):
    """Determine target pollutant with fallback handling"""
    target = ALTERNATE_TARGETS.get(state, 'PM2.5 (ug/m3)')
    
    if target not in df.columns:
        for strategy in FALLBACK_STRATEGIES.get(target, []):
            try:
                df[target] = strategy(df)
                if not df[target].isnull().all():
                    return target
            except KeyError:
                continue
        raise ValueError(f"No fallback strategy worked for {state}")
    return target

def train_state_model(state):
    """Train and save XGBoost model for a specific state"""
    try:
        #load preprocessed data
        file_path = f"{STATES_DIR}/{state}_processed.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed data not found: {file_path}")
            
        df = pd.read_csv(
            file_path,
            parse_dates=['datetime'],
            index_col='datetime',
            dtype={'PM2.5 (ug/m3)': 'float32', 'PM10 (ug/m3)': 'float32'},
            low_memory=False
        )
        
        #find target pollutant
        target = get_target(state, df)
        print(f"Using target '{target}' for {state}")
        
        #Feature engineering
        df = create_features(df)
        print("Features created")
        df = create_lag_features(df,target)
        print("Lag Features created")
        print(f"Target for imputation: {target}")
        df = impute_missing_lags(df, ['pm_lag_1Y', 'pm_lag_2Y'], target)
        print("missing lags imputed")
        
        #select features based on availability
        available_features = [
            'hour', 'month', 'year', 'pm_lag_1Y', 'pm_lag_2Y',
            'CO (mg/m3)', 'Ozone (ug/m3)', 'NOx (ug/m3)', 'SO2 (ug/m3)'
        ]
        features = [f for f in available_features if f in df.columns]
        
        #Train/test split (temporal)
        train_size = int(len(df) * 0.8)
        X_train = df[features].iloc[:train_size]
        y_train = df[target].iloc[:train_size]
        
        #Drop rows where target is NaN or infinite
        valid_mask = y_train.notna() & y_train.apply(lambda x: np.isfinite(x))
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        # Train model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=18
        )
        model.fit(X_train, y_train)
        print("model is fit")
        # Save model
        os.makedirs(MODELS_DIR, exist_ok=True)
        model.save_model(f"{MODELS_DIR}/xgboost_{state}_model.json")
        
        # Save feature list
        with open(f"{MODELS_DIR}/{state}_features.txt", "w") as f:
            f.write("\n".join(features))
            
        print(f"✅ Model saved for {state} | Features: {features}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {state}: {str(e)}")
        return False

if __name__ == "__main__":
    state_files = [f.replace('_processed.csv', '') 
                   for f in os.listdir(STATES_DIR) 
                   if f.endswith('_processed.csv')]
    
    print(f"Found {len(state_files)} states to process")
    
    for state in state_files:
        print(f"\n{'='*50}")
        print(f"Training model for {state}...")
        success = train_state_model(state)
        if success:
            print(f"Completed {state} successfully")
        else:
            print(f"Failed to process {state}")
