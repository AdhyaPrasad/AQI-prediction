import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns             
import os                          
import glob                        
import time 
import streamlit as st

COLOR_SCHEME = {
    "navy": "#0a1128",
    "red": "#E93F33",
    "yellow": "#FFF833",
    "green": "#55A84F",
    "light-green": "#A3C853",
    "light": "#f8f9fa",
    "dark-red":"#AF2D24",
    "orange":"#F29C33",
    "purple":"#C697CA",
    "darkbrown":"#523E43",
    "btn-color":"#3cb2d0",
    "azul": "#226FAF",
    "verdigris": "#72B7B2",
    "seasalt": "#F8F9F9",
    "tea_green": "#DAF1B5",
    "marian_blue": "#24489F",
    "nyanza": "#D7E9CA",
    "timberwolf": "#D1CDCC",
    "denim": "#2F5BA3",
    "blue_ncs": "#2D8AB7",
    "polynesian_blue": "#23489A"


}
def preprocess_air_quality_data(df):
    """
    Cleans and preprocesses air quality dataframe:
    - Converts 'From Date' to datetime and sets as index
    - Drops 'To Date'
    - Merges similar columns based on a predefined mapping

    Parameters
    ----------
    df : pd.DataFrame
        Raw air quality dataframe

    Returns
    -------
    df : pd.DataFrame
        Preprocessed and cleaned dataframe
    """
    #HOW TO USE IT
    # Preprocess a state's data
    #df = preprocess_air_quality_data(all_states_data['Delhi'])
    #df.head()
    
    # Step 1: Datetime Index
    df = df.drop(columns='To Date', errors='ignore')
    df['From Date'] = pd.to_datetime(df['From Date'], errors='coerce')
    df = df.rename(columns={'From Date': 'datetime'})
    df = df.set_index('datetime')

    # Step 2: Column Merging
    reduction_groups = {
        "Xylene (ug/m3)":    ["Xylene ()"],
        "MP-Xylene (ug/m3)": ["MP-Xylene ()"],
        "Benzene (ug/m3)":   ["Benzene ()"],
        "Toluene (ug/m3)":   ["Toluene ()"],
        "SO2 (ug/m3)":       ["SO2 ()"],
        "NOx (ug/m3)":       ["NOx (ppb)"],
        "Ozone (ug/m3)":     ["Ozone (ppb)"],
        "AT (degree C)":     ["AT ()"],
        "WD (degree)":       ["WD (degree C)", "WD (deg)", "WD ()"],
        "WS (m/s)":          ["WS ()"]
    }

    for column, cols_to_merge in reduction_groups.items():
        if column not in df.columns and any(name in df.columns for name in cols_to_merge):
            df[column] = np.nan

        for col_name in cols_to_merge:
            if col_name in df.columns:
                df[column] = df[column].fillna(df[col_name])
                df = df.drop(columns=[col_name])

    return df



# %%
def clean_missing_values(df, threshold=0.6, plot_missing=True):
    """
    Cleans a DataFrame by:
    - Dropping completely empty rows and columns
    - Dropping columns with more than `threshold` missing values
    - Returning cleaned df and missing value report
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean.
    threshold : float
        Minimum fraction of non-null values a column must have (default 0.6 = 60% non-null).
    plot_missing : bool
        Whether to show a barplot of missing value percentages.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame.
    df_null_info : pd.DataFrame
        DataFrame with null count and percentage info for each column.
    """

    # Drop rows and columns that are completely NaN
    df = df.dropna(how='all')                  # Drop empty rows
    df = df.dropna(how='all', axis=1)          # Drop empty columns

    # Null info before thresholding
    null_vals = df.isnull().sum()
    df_null_info = pd.concat({
        'Null Count': null_vals,
        'Percent Missing (%)': round(null_vals * 100 / len(df), 2)
    }, axis=1).sort_values(by='Null Count', ascending=False)

    #drop columns that exceed the missing value threshold
    df = df.dropna(thresh=int(df.shape[0] * threshold), axis=1)

    if plot_missing:
        plt.figure(figsize=(8, max(6, 0.3 * len(df_null_info))))
        sns.barplot(data=df_null_info, x='Percent Missing (%)', y=df_null_info.index, orient='h', color='steelblue')
        plt.title("Missing Value Percentage per Feature")
        plt.show()

    return df, df_null_info

def create_lag_features(df,target):
    df = df.copy()
    df['pm_lag_1Y'] = df[target].shift(365 * 24)  # 1 year lag
    df['pm_lag_2Y'] = df[target].shift(730 * 24)  # 2 year lag
    return df


def impute_lag_with_monthly_avg(df, lag_col, ref_col):
    """
    Replaces NaNs in the lag feature using the monthly average of the reference column.

    Parameters:
    - df (DataFrame): Input time-indexed DataFrame.
    - lag_col (str): The lag feature with NaNs (e.g., 'pm_lag_1Y').
    - ref_col (str): The original column to compute monthly means from (e.g., 'PM2.5 (ug/m3)').

    Returns:
    - df (DataFrame): Updated DataFrame with NaNs in lag_col replaced by same-month averages.
    """
    df = df.copy()
    
    for month in range(1, 13):
        month_mask = df.index.month == month
        monthly_avg = df.loc[month_mask, ref_col].mean()
        
        #replacing NaNs in lag_col for that month
        condition = df[lag_col].isna() & month_mask
        df.loc[condition, lag_col] = monthly_avg

    return df

def impute_missing_lags(df, lag_columns, ref_col='PM2.5 (ug/m3)'):
    for lag_col in lag_columns:
        df = impute_lag_with_monthly_avg(df, lag_col, ref_col)
    return df

def create_features(df, required_features=None):
    """Create datetime-based features matching training features"""
    df = df.copy()
    
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    #create other features if they were used in training
    if required_features:
        if 'dayofmonth' in required_features:
            df['dayofmonth'] = df.index.day
        if 'dayofweek' in required_features:
            df['dayofweek'] = df.index.dayofweek
        if 'dayofyear' in required_features:
            df['dayofyear'] = df.index.dayofyear
        if 'weekofyear' in required_features:
            df['weekofyear'] = df.index.isocalendar().week.astype("int64")
        if 'quarter' in required_features:
            df['quarter'] = df.index.quarter
    
    return df


def add_pollutant_features(df, historical_data):
    """
    Add pollutant features with patterns EXTRACTED FROM HISTORICAL DATA
    """
    df = df.copy()
    pollutants = ['CO (mg/m3)', 'Ozone (ug/m3)', 'NOx (ug/m3)', 'SO2 (ug/m3)']
    
    for pollutant in pollutants:
        if pollutant in historical_data.columns:
            # 1. Extract REAL seasonal pattern
            seasonal_pattern = historical_data.groupby(historical_data.index.month)[pollutant].mean()
            
            # 2. Extract REAL daily pattern
            daily_pattern = historical_data.groupby(historical_data.index.hour)[pollutant].mean()
            
            # 3. Historical statistics
            hist_std = historical_data[pollutant].std()
            hist_mean = historical_data[pollutant].mean()
            
            for idx in df.index:
                month = idx.month
                hour = idx.hour
                
                # Base value from ACTUAL monthly pattern
                base_value = seasonal_pattern.get(month, hist_mean)
                
                # Apply REAL daily variation
                daily_factor = daily_pattern.get(hour, hist_mean) / hist_mean
                
                # Add noise based on ACTUAL variability
                noise = np.random.normal(0, hist_std * 0.2)  # 20% of std dev
                
                final_value = base_value * daily_factor + noise
                df.loc[idx, pollutant] = max(0, final_value)
    
    return df

def get_health_category(aqi):
    aqi=float(aqi)
    if aqi <= 50:
        return "Good", COLOR_SCHEME['green']
    elif aqi <= 100:
        return "Satisfactory", COLOR_SCHEME['light-green']
    elif aqi <= 200:
        return "Moderate", COLOR_SCHEME['yellow']
    elif aqi <= 300:
        return "Poor", COLOR_SCHEME['orange']
    elif aqi <= 400:
        return "Very Poor", COLOR_SCHEME['red']
    else:
        return "Severe", COLOR_SCHEME['dark-red']


def create_monthly_summary(predictions):
    monthly_avg = predictions.resample('M').mean()
    
    monthly_data = pd.DataFrame({
        'Month': monthly_avg.index.strftime('%B'),
        'Mean AQI': monthly_avg.apply(lambda x: f"{x:.2f}"),
        'Health Risk': '',
        'Color': ''
    })
    
    health_info = monthly_avg.apply(lambda x: pd.Series(get_health_category(x)))
    monthly_data[['Health Risk', 'Color']] = health_info.values
    
    return monthly_data



def create_future_dataset_direct(raw_data, target, start_date, end_date, required_features):
   
    future_df = pd.DataFrame(
        pd.date_range(start=start_date, end=end_date, freq='1h'),
        columns=['datetime']
    ).set_index('datetime')
    
    future_df = create_features(future_df, required_features)
    
    lag_1y = future_df.index - pd.Timedelta('365 days')
    lag_2y = future_df.index - pd.Timedelta('730 days')
    
    future_df['pm_lag_1Y'] = [
        raw_data[target].get(ts, raw_data[target].mean())
        for ts in lag_1y
    ]
    future_df['pm_lag_2Y'] = [
        raw_data[target].get(ts, raw_data[target].mean())
        for ts in lag_2y
    ]
    
    for feature in required_features:
        if feature not in future_df.columns:
            if feature in raw_data.columns:
                #historical mean for missing features
                future_df[feature] = raw_data[feature].mean()
            else:
                #fallback to 0 if feature unknown
                future_df[feature] = 0
                
    return future_df



#not used
def get_health_risk(aqi):
    """Map AQI values to health risk categories"""
    if aqi <= 50:
        return "Good (Minimal health risk)"
    elif aqi <= 100:
        return "Moderate (May cause respiratory symptoms in sensitive individuals)"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups (May cause breathing difficulty)"
    elif aqi <= 200:
        return "Unhealthy (May cause breathing difficulty to general population)"
    elif aqi <= 300:
        return "Very Unhealthy (May cause respiratory illness on prolonged exposure)" 
    else:
        return "Hazardous (Serious risk of respiratory effects)"
