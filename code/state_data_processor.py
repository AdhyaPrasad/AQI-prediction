import os
import pandas as pd
from tqdm import tqdm
from aqi_functions import preprocess_air_quality_data,clean_missing_values
RAW_DATA_DIR = "data/states_combined"
PROCESSED_DIR = "data/processed_states"

def preprocess_and_save(state_name):
    try:
        raw_df = pd.read_csv(f"{RAW_DATA_DIR}/{state_name}.csv", low_memory=False)
        
        df = preprocess_air_quality_data(raw_df)
        df, _ = clean_missing_values(df, threshold=0.6, plot_missing=False)
        
        df.to_csv(f"{PROCESSED_DIR}/{state_name}_processed.csv", index=True)
        return True
    except Exception as e:
        print(f"Error processing {state_name}: {str(e)}")
        return False

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    state_files = [f.replace('.csv', '') 
                   for f in os.listdir(RAW_DATA_DIR) 
                   if f.endswith('.csv')]
    
    for state in tqdm(state_files, desc="Preprocessing states"):
        preprocess_and_save(state)
