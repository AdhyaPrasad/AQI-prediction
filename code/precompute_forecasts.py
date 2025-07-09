import pandas as pd
import sqlite3
import os
import xgboost as xgb
from tqdm import tqdm
from aqi_functions import create_future_dataset_direct
from state_modeling import get_target


STATES_DIR = "data/processed_states"
MODELS_DIR = "models/state_models"
DB_PATH = "precomputed_forecasts.db"
TABLE_NAME = "forecasts"
YEARS = range(2024, 2026)

def get_valid_states():
    valid_states = []
    for state_file in os.listdir(STATES_DIR):
        if state_file.endswith('_processed.csv'):
            state = state_file.replace('_processed.csv', '')
            model_path = f"{MODELS_DIR}/xgboost_{state}_model.json"
            if os.path.exists(model_path):
                valid_states.append(state)
    return valid_states

def fix_dtypes(df):
    """Convert lag columns to float and handle missing values"""
    for col in ['pm_lag_1Y', 'pm_lag_2Y']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].mean(), inplace=True)
    return df

def verify_db_setup():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='forecasts';")
    if not cursor.fetchone():
        create_db_table()
    conn.close()

def create_db_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT NOT NULL,
            year INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            prediction REAL NOT NULL,
            UNIQUE(state, year, timestamp)
        )
    ''')
    conn.commit()
    conn.close()

def save_predictions(state, year, predictions, timestamps):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    data = [(state, year, str(ts), float(pred)) 
            for ts, pred in zip(timestamps, predictions)]
    
    cursor.executemany(
        f'''INSERT OR IGNORE INTO {TABLE_NAME} 
        (state, year, timestamp, prediction) 
        VALUES (?, ?, ?, ?)''', data)
    
    conn.commit()
    conn.close()

def load_model(state):
    model = xgb.XGBRegressor()
    model.load_model(f"{MODELS_DIR}/xgboost_{state}_model.json")
    return model

def load_data(state):
    """Load data with dtype fixes"""
    df = pd.read_csv(
        f"{STATES_DIR}/{state}_processed.csv", 
        parse_dates=['datetime'], 
        index_col='datetime',
        low_memory=False
    )
    df = fix_dtypes(df)  
    return df

def precompute_all():
    verify_db_setup() 
    create_db_table()
    states = get_valid_states()  
    
    for state in tqdm(states, desc="Processing states"):
        try:
            model = load_model(state)
            df = load_data(state)
            
            with open(f"{MODELS_DIR}/{state}_features.txt", "r") as f:
                features = f.read().splitlines()
            target = get_target(state, df)
            
            for year in YEARS:
                future_df = create_future_dataset_direct(
                    raw_data=df,
                    target=target,
                    start_date=f"{year}-01-01",
                    end_date=f"{year}-12-31 23:00:00",
                    required_features=features
                )
                future_df = future_df.apply(pd.to_numeric, errors='coerce')
                predictions = model.predict(future_df)
                save_predictions(state, year, predictions, future_df.index)
                
        except Exception as e:
            print(f"Error processing {state} ({year}): {str(e)}")
    
    print("Precomputation complete!")

if __name__ == "__main__":
    precompute_all()
