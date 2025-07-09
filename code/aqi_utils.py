# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
import os                          
import glob                        
import time                       
import numpy as np                 
import pandas as pd                
import matplotlib.pyplot as plt   
import seaborn as sns             
sns.set_theme()

# sklearn imports
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import (
    cross_val_score,
    TimeSeriesSplit,
    RandomizedSearchCV
)

import xgboost as xgb                       
from IPython.display import clear_output   

# %%
N_JOBS = -1
# Random variable for having consistent results between runs
RANDOM_STATE = 18
#DATASET_SRC = '/kaggle/input/time-series-air-quality-data-of-india-2010-2023'
DATASET_SRC='data'
df_states = pd.read_csv(f'{DATASET_SRC}/stations_info.csv')
df_states.drop(columns=['agency', 'station_location', 'start_month'], inplace=True)
print(df_states.head())

# %%
unique_states = df_states['state'].unique()
unique_states


# %%
def combine_state_df(state_name):
    """
    Combine all CSV files for a given state into one dataframe.
    Adds the city name based on df_states mapping.
    """
    state_code = df_states[df_states['state'] == state_name]['file_name'].iloc[0][:2]
    state_files = glob.glob(f'{DATASET_SRC}/{state_code}*.csv')
    
    print(f'Combining {len(state_files)} files for {state_name}...\n')

    combined_df = []

    for file in state_files:
        file_name = os.path.basename(file).replace('.csv', '')
        df = pd.read_csv(file)
        city = df_states[df_states['file_name'] == file_name]['city'].values[0]
        df['city'] = str(city)
        combined_df.append(df)

    return pd.concat(combined_df, ignore_index=True)

def get_all_combined_states_data():
    """
    Combines all state-level CSV files into separate dataframes,
    stored in a dictionary with state names as keys.

    Returns
    -------
    state_dfs (dict): Dictionary with keys as state names and values as combined DataFrames
    """
    state_dfs = {}
    unique_states = df_states['state'].unique()

    for state in unique_states:
        try:
            print(f'Processing: {state}')
            state_df = combine_state_df(state)
            state_dfs[state] = state_df
            print(f'DataFrame stored for {state} ✅\n')
        except Exception as e:
            print(f'❌ Error processing {state}: {e}\n')
    
    return state_dfs

all_states_data = get_all_combined_states_data()



# %%
karnataka_df = all_states_data['Karnataka']
print(karnataka_df.info())


# %%
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

    # Drop columns that exceed the missing value threshold
    df = df.dropna(thresh=int(df.shape[0] * threshold), axis=1)

    # Optional visualization
    if plot_missing:
        plt.figure(figsize=(8, max(6, 0.3 * len(df_null_info))))
        sns.barplot(data=df_null_info, x='Percent Missing (%)', y=df_null_info.index, orient='h', color='steelblue')
        plt.title("Missing Value Percentage per Feature")
        plt.show()

    return df, df_null_info



# %%
# Step 1: Preprocess Delhi data
if __name__=="__main__":
    delhi_raw_df = all_states_data['Delhi']
    delhi_df = preprocess_air_quality_data(delhi_raw_df)
    delhi_df, delhi_null_info = clean_missing_values(delhi_df, threshold=0.6)


# %% [markdown]
# **Karnataka's df to compare**

# %%
    ktaka_raw_df = all_states_data['Karnataka']
    ktaka_df = preprocess_air_quality_data(ktaka_raw_df)
    ktaka_df, ktaka_null_info = clean_missing_values(ktaka_df, threshold=0.6)

# %% [markdown]
# Different states have collected varying types and quantities of air quality metrics. In datasets like this, it's common practice to retain features that have less than 25–30% missing data, unless the feature holds critical importance. For our analysis, we chose to drop columns in datasets that have more than 40% missing values to ensure data quality, thats why we chose a threshold of 0.6 beforehand.

# %% [markdown]
# # Exploratory Data Analysis(EDA)
# I'm focusing on Delhi's AQI, its analysis and and forecasting.

# %%
pollutants = {
    # A mixture of solid particles and liquid droplets found in the air.
    'Particulate Matter' : ['PM2.5 (ug/m3)', 'PM10 (ug/m3)'],

    # Nitrogen gases form when fuel is burned at high temperatures.
    'Nitrogen Compounds' : ['NOx (ug/m3)', 'NO (ug/m3)', 'NO2 (ug/m3)', 'NH3 (ug/m3)'],

    # These are found in coal tar, crude petroleum, paint, vehicle exhausts and industrial emissions.
    'Hydrocarbons' : ['Benzene (ug/m3)', 'Eth-Benzene (ug/m3)', 'Xylene (ug/m3)', 'MP-Xylene (ug/m3)', 'O Xylene (ug/m3)', 'Toluene (ug/m3)'],

    # Released from the partial combustion of carbon-containing compounds.
    'Carbon Monoxide': ['CO (mg/m3)'],

    # Released naturally by volcanic activity and is produced as a by-product of copper extraction and the burning of sulfur-bearing fossil fuels.
    'Sulfur Dioxide': ['SO2 (ug/m3)'],

    # It is created mostly the combustion of fossil fuels.
    'Ozone Concentration' : ['Ozone (ug/m3)']
}

other_metrics = {
    # Affects Earth's average temperatures
    'Solar Radiation' : ['SR (W/mt2)'],

    'Temperatures' : ['Temp (degree C)', 'AT (degree C)'],

    'Relative Humidity' : ['RH (%)'],

    'Rainfall' : ['RF (mm)'],

    'Barometric Pressure' : ['BP (mmHg)'],

    'Wind Direction' : ['WD (degree)'],

    'Wind Speed' : ['WS (m/s)']
}


# %% [markdown]
# Some may have been dropped due to missing values, so a couple functions to check their presence

# %%
available_pollutants = {
    group: [col for col in cols if col in delhi_df.columns]
    for group, cols in pollutants.items()
}

available_other_metrics = {
    group: [col for col in cols if col in delhi_df.columns]
    for group, cols in other_metrics.items()
}


# %% [markdown]
# # Time Frequencies
# Let's start by grouping our DataFrame by various frequencies.

# %%
slice_groups = {
    'Group by Day':   delhi_df.groupby(pd.Grouper(freq='1D')).mean(numeric_only=True),
    'Group by Month': delhi_df.groupby(pd.Grouper(freq='1ME')).mean(numeric_only=True),
    'Group by Year':  delhi_df.groupby(pd.Grouper(freq='1YE')).mean(numeric_only=True)
}


# %%
def plot_features_by_group(features, slice_groups):    
    for feature in features:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        fig.suptitle(feature)
        
        labels = []
        for i, (group, group_df) in enumerate(slice_groups.items()):
            data_slice = group_df[group_df.columns.intersection(pollutants[feature])]
            
            # Keep only the NOx feature, as it combines both NO (Nitrogen Oxide) and NO2 (Nitrogen Dioxide)
            if feature == "Nitrogen Compounds":
                data_slice = data_slice.drop(['NO (ug/m3)', 'NO2 (ug/m3)'], axis=1)
                
            data_slice.plot(kind="line", ax=ax)
            
            for column in data_slice.columns:
                labels.append(f'{column} [{group}]')
        
        ax.set(xlabel=None)
        ax.legend(labels)
        plt.plot()


# %%
features_to_plot = ['Particulate Matter', 'Carbon Monoxide', 'Ozone Concentration', 'Nitrogen Compounds']
plot_features_by_group(features_to_plot, slice_groups)

# %% [markdown]
# # Seasonal Boxplots by Month

# %%
delhi_df['month'] = delhi_df.index.month
sns.boxplot(data=delhi_df, x='month', y='PM2.5 (ug/m3)')
plt.title("Monthly Distribution of PM2.5 in Delhi")
plt.show()


# %% [markdown]
# * Nov–Jan: Severe pollution	(Crop burning, festivals, low dispersion, cold air)
# * Jun–Sep: Low pollution	(Rainfall, fewer emissions)
# * Oct, Feb: Transition periods

# %%
delhi_df['month'] = delhi_df.index.month
sns.boxplot(data=delhi_df, x='month', y='Ozone (ug/m3)')
plt.title("Monthly Distribution of Ozone Concentration in Delhi")
plt.show()

# %% [markdown]
# ### Air Quality Impact During COVID-19 Lockdown in Delhi (Mar–May 2020)

# %%
for feature in features_to_plot:
    data_slice = slice_groups['Group by Day'][slice_groups['Group by Day'].columns.intersection(pollutants[feature])]
    data_slice.loc['2020-02':'2020-06'].plot(title=f'{feature} (Feb–Jun 2020)', figsize=(12, 4)).set(xlabel=None)


# %% [markdown]
# ## Pair plot
# We can see a better explanation on the relationships between the variables, as well as the distribution of each one through a pair plot.

# %%
sns.pairplot(slice_groups['Group by Month'])
plt.suptitle("Pair Plot - Monthly Averaged Pollutants", y=1.02)
plt.show()

# %% [markdown]
# * Strong relationships between key pollutants (PM2.5 ↔ NOx/NO2)
# * Some features like ozone act independently

# %% [markdown]
# ##  Correlation Matrix
# Correlation matrix helps easily visuallize the correlation degree between the variables.

# %%
corr = slice_groups['Group by Day'].corr(numeric_only=True).round(2)
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10,5))
sns.heatmap(data=corr, mask=mask, annot=True, cmap="rocket_r")
plt.title('Daily Correlation Map')
plt.show()

# %%
corr_month = slice_groups['Group by Month'].corr(numeric_only=True).round(2)
mask = np.triu(np.ones_like(corr_month, dtype=bool))

plt.figure(figsize=(10,5))
plt.title('Monthly Correlation Map')

sns.heatmap(data=corr_month, mask=mask, annot=True, cmap="rocket_r")
plt.show()

# %% [markdown]
# ## Correlation Insights: Daily vs Monthly Heatmaps
#
# #### Daily Correlation Map
# - **Strong correlations** observed between:
#   - `NO` and `NO₂` (**0.69**): Indicates their co-emission from common sources like vehicles and industries.
#   - `PM2.5` and `NO` / `NOx`: Suggests particulate matter is heavily influenced by nitrogen-based emissions on a daily scale.
# - **Ozone** shows **very weak or slightly negative correlations** with nitrogen compounds and particulate matter.
#   - This aligns with ozone’s nature as a **secondary pollutant** (forms photochemically, not directly emitted).
# - **Carbon Monoxide (CO)** shows low correlation with others, indicating **independent variability** in short-term emissions.
#
# #### Monthly Correlation Map
# - Similar strong trends persist between `NO`, `NO₂`, and `NOx`, but correlation values are even **stronger** due to **seasonal averaging**:
#   - `NO` ↔ `NOx`: **0.85**
#   - `NO₂` ↔ `NOx`: **0.86**
# - `PM2.5` correlations with NO-based compounds also **strengthen**, highlighting **seasonal pollution patterns** (e.g., winter spikes).
# - Ozone shows a **slightly stronger negative correlation** with nitrogen oxides, reflecting the **inverse seasonal relationship**:
#   - More ozone in summer (due to photochemical activity), less when NOx is high (winter).
#
# Overall, monthly aggregation reveals **clearer seasonal trends** and **smoother pollutant relationships**, while daily correlations highlight **short-term co-fluctuations** from local emission events.
#

# %%
corr_target = abs(corr['PM2.5 (ug/m3)'])
relevant_features = corr_target[corr_target>0.4]
relevant_features.sort_values(ascending=False)

# %% [markdown]
# Both NO2 and NOx are highly correlated with PM2.5 and can serve as strong predictors in your model
# These are most likely co-emitted from combustion sources like traffic, industries, biomass burning

# %% [markdown]
# ## Trend decomposition
# It separates a time series into trend, seasonality, and noise components.
# It helps uncover long-term patterns and recurring seasonal effects in pollutants like PM2.5 or Ozone.
#

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
# Group by day to smooth out noise
pm_daily = delhi_df['PM2.5 (ug/m3)'].resample('D').mean()

# Drop missing values
pm_daily = pm_daily.dropna()
# Perform seasonal decomposition
result = seasonal_decompose(pm_daily, model='additive', period=365)  # yearly seasonality

# Plot the components
result.plot()
plt.suptitle("Seasonal Decomposition of PM2.5 in Delhi", fontsize=16)
plt.tight_layout()
plt.show()


# %% [markdown]
# - The decomposition reveals strong seasonal patterns in PM2.5, with peaks consistently occurring during winter months and troughs during the monsoon.
# - Long-term trend shows a rise in pollution until 2012, followed by a gradual decrease or stabilization, possibly due to regulatory actions or policy shifts.
# - Residuals confirm the presence of unexpected pollution events, highlighting the impact of irregular or one-time incidents.
#

# %% [markdown]
# ## Feature Engineering
# ### Drop Correlated Features
# Both NO and NO₂ are highly correlated with NOx, because:
#
# NOx = NO + NO₂ (i.e., it's already a combined measure)
#
# Including all three would create redundancy in your dataset

# %%
# Drop correlated nitrogen compound features
#delhi_df = delhi_df.drop(['NO (ug/m3)', 'NO2 (ug/m3)'], axis=1)
#delhi_df.head()
delhi_df = delhi_df.drop(columns=[col for col in ['NO (ug/m3)', 'NO2 (ug/m3)'] if col in delhi_df.columns])


# %% [markdown]
# ### Resampling
# Resampling is a method used in time series data to change the frequency of your observations
# Secondly, this combined dataframe can contain data for the same timeframe as measurements ware made from various locations within the state. Here as I am interested in exploring the air quality in one state at a time, I will resample the same datetime measurements by taking the **mean** of the measurements.

# %%
delhi_df = delhi_df.resample('60min').mean(numeric_only=True)

# %% [markdown]
# ### Outlier Detection and Removal
# In general outliers are able to distort analyses and skew results. They are extreme values that can greatly differ from the rest of the data. By removing the influence of such extreme data points we can make more robust and accurate predictions.

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.tight_layout(pad=3.0)

sns.histplot(data=delhi_df['PM2.5 (ug/m3)'], bins=250, kde=True, ax=axes[0,0])
sns.histplot(data=delhi_df['CO (mg/m3)'], bins=250, kde=True, ax=axes[0,1])
sns.histplot(data=delhi_df['Ozone (ug/m3)'], bins=250, kde=True, ax=axes[1,0])
sns.histplot(data=delhi_df['NOx (ug/m3)'], bins=250, kde=True, ax=axes[1,1])

plt.show()


# %% [markdown]
# The first feature we will explore is the Particulate Matter (PM2.5).

# %%
delhi_df.query('`PM2.5 (ug/m3)` > 600')['PM2.5 (ug/m3)'].plot(style='.', figsize=(10,4))

# %% [markdown]
# Here we can probably notice that we have just a few outliers above 950 around the year of 2012. I am going to remove them with caution.

# %%
delhi_df['PM2.5 (ug/m3)'] = delhi_df['PM2.5 (ug/m3)'].mask(delhi_df['PM2.5 (ug/m3)'].gt(950))
delhi_df.query('`PM2.5 (ug/m3)` > 600')['PM2.5 (ug/m3)'].plot(style='.', figsize=(10,4))

# %% [markdown]
# The next feature is Carbon Monoxide(CO).

# %%
delhi_df.query('`CO (mg/m3)` > 20')['CO (mg/m3)'].plot(style='.', figsize=(10,4))

# %% [markdown]
# This feature is quite noisy. However there is a group on the top-right side of the plot and after the year 2015. 

# %%
delhi_df['CO (mg/m3)'] = delhi_df['CO (mg/m3)'].mask(((delhi_df.index > '2015') & delhi_df['CO (mg/m3)'].gt(35)))
delhi_df.query('`CO (mg/m3)` > 20')['CO (mg/m3)'].plot(style='.', figsize=(10,4))

# %% [markdown]
# Next feature is Ozone.

# %%
delhi_df.query('`Ozone (ug/m3)` > 140')['Ozone (ug/m3)'].plot(style='.', figsize=(10,4))

# %%
delhi_df['Ozone (ug/m3)'] = delhi_df['Ozone (ug/m3)'].mask(delhi_df['Ozone (ug/m3)'].gt(185))


# %% [markdown]
# Lastly we take a look at the Nitrogen Compounds (NOx) feature.

# %%
delhi_df.query('`NOx (ug/m3)` > 350')['NOx (ug/m3)'].plot(style='.', figsize=(10,4))


# %% [markdown]
# Again, we notice just a few extreme points that may be error data points. I will eliminate those.

# %%
delhi_df['NOx (ug/m3)'] = delhi_df['NOx (ug/m3)'].mask((
    ((delhi_df.index < '2013') & (delhi_df['NOx (ug/m3)'].gt(380))) |
    ((delhi_df.index > '2015') & (delhi_df.index < '2016') & (delhi_df['NOx (ug/m3)'].gt(400))) |
    ((delhi_df.index > '2016') & (delhi_df['NOx (ug/m3)'].gt(450)))
))


# %% [markdown]
# ### Handling Missing Values

# %%
def get_null_info(dataframe):
    null_vals = dataframe.isnull().sum()
    df_null_vals = pd.concat({
        'Null Count': null_vals,
        'Percent Missing (%)': round(null_vals * 100 / len(dataframe), 2)
    }, axis=1)

    return df_null_vals.sort_values(by='Null Count', ascending=False)



# %%
get_null_info(delhi_df)

# %%
delhi_df = delhi_df.ffill()
# Fill any remaining NaNs with column means (if any)
delhi_df = delhi_df.fillna(delhi_df.mean(numeric_only=True))
delhi_df.info()


# %% [markdown]
# ### Date Component Features
# Let's prepare our dataset by enhancing it with useful features and separating it into training/testing splits.

# %%
def create_features(df):
    df = df.copy()
    df['hour']       = df.index.hour
    df['dayofmonth'] = df.index.day
    df['dayofweek']  = df.index.dayofweek
    df['dayofyear']  = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype("int64")
    df['month']      = df.index.month
    df['quarter']    = df.index.quarter
    df['year']       = df.index.year
    return df



# %%
# Apply to your cleaned dataset
delhi_df = create_features(delhi_df)

# List of new time-based features (optional reference)
date_features = ['hour', 'dayofmonth', 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'quarter', 'year']


# %% [markdown]
# Now it is very easy to visualize the various metrics by the above features, like through boxplots. 
# Example: check the air quality through the months.

# %%
def plot_by_datetime(df, metric, time_groups):
    for time_group in time_groups:
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.boxplot(data=df, x=time_group, y=metric, palette="icefire", showfliers=False)
        ax.set_title(f'{metric} by {time_group}')
        ax.set(xlabel=time_group)
        plt.tight_layout()
        plt.show()



# %%
plot_by_datetime(delhi_df, 'PM2.5 (ug/m3)', ['hour', 'dayofmonth', 'dayofweek', 'weekofyear', 'month', 'quarter', 'year'])


# %% [markdown]
# These plots indicate that the various datetime groups capture important trends and seasonal patterns in PM2.5 levels. Notably, features like `hour`, `month`, and `quarter` show strong variability, suggesting they hold predictive value for modeling air quality.
# Interestingly, the `dayofweek` feature appears to have a fairly uniform distribution across all days, indicating it may be less informative on its own. Nevertheless, we will include all extracted datetime features in the model to allow it to learn any subtle patterns or interactions that may exist.

# %% [markdown]
# ## Lag Features
# Lag features capture information about a variable in a prior time step. In the case of forecasting, such lag features are likely to be predictive and help our models. What's more, we can also include lag features based on other predictive features in order to improve the forecasting accuracy.

# %%
def create_lag_features(df):
    df = df.copy()
    df['pm_lag_1Y'] = df['PM2.5 (ug/m3)'].shift(365 * 24)  # 1 year lag
    df['pm_lag_2Y'] = df['PM2.5 (ug/m3)'].shift(730 * 24)  # 2 year lag
    return df

lag_features = ['pm_lag_1Y', 'pm_lag_2Y']
delhi_df = create_lag_features(delhi_df)
delhi_df.head()


# %% [markdown]
# What Lag Features Do:
# They let the model "look back in time" to learn:
#
# What happened last year at this hour (pm_lag_1Y)
# What happened two years ago (pm_lag_2Y)
#
# This is particularly helpful because PM2.5 levels have strong seasonality, and past values at the same time of year can be highly predictive.
#
#

# %% [markdown]
# ### Handling Missing Lag Values
#
# After generating lag features (e.g., PM2.5 from last year), the earliest records naturally contain missing values. These can pose issues during modeling, especially for algorithms like Random Forest or XGBoost that don’t handle NaNs.
#
# To avoid introducing bias (e.g., by filling with zeros), I chose to replace lag NaNs with values based on the same time (e.g., same month/day in other years). This is a seasonal-aware imputation strategy.
#
#

# %%
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
        
        # Replace NaNs in lag_col for that month
        condition = df[lag_col].isna() & month_mask
        df.loc[condition, lag_col] = monthly_avg

    return df



# %%
# Apply to both lag features
delhi_df = impute_lag_with_monthly_avg(delhi_df, 'pm_lag_1Y', 'PM2.5 (ug/m3)')
delhi_df = impute_lag_with_monthly_avg(delhi_df, 'pm_lag_2Y', 'PM2.5 (ug/m3)')
delhi_df.head()


# %% [markdown]
# ## Time Series Forecasting
# Note: Could have avoided filling NaN values by using XGBoost

# %%
def replace_lag_na(df, how):
    '''
    Replaces missing values by applying various methods.
    
    Some additional ideas to implement include:
      1. Replace lag NaNs with the overall chosen method for that variable
      2. Replace lag NaNs with the time chosen method for the variable in the window value
    '''

    # Replace lag NaNs with zeros
    if how == 'zeros':
        return df.fillna(0)
    # Drop missing lag records
    if how == 'drop':
        return df.dropna(how='any')


# %%
target = 'PM2.5 (ug/m3)'
predictors = date_features + lag_features

def create_train_test_sets(dataframe, split=0.8, replace_na=False, method='none'):
    """
    Creates train and test sets with optional handling of missing values.

    Parameters:
        dataframe (DataFrame): Input data with predictors + target.
        split (float): Fraction of data to use for training.
        replace_na (bool): Whether to handle missing values.
        method (str): How to handle missing values ('drop' or 'zeros').

    Returns:
        X_train, X_test, y_train, y_test
    """
    dataframe = dataframe.copy()

    if replace_na:
        dataframe = replace_lag_na(dataframe, how=method)

    train_set, test_set = np.split(dataframe, [int(len(dataframe) * split)])

    return train_set[predictors], test_set[predictors], train_set[target], test_set[target]



# %%
X_train, X_test, y_train, y_test = create_train_test_sets(delhi_df, split=0.8, replace_na=True, method='drop')

# %%
ensemble_models = {
    'Random Forest':     RandomForestRegressor(random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
    'AdaBoost':          AdaBoostRegressor(random_state=RANDOM_STATE),
    'Histogram GB':      HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    'XGBoost':           xgb.XGBRegressor(random_state=RANDOM_STATE)
}


# %% [markdown]
# I am going to use various metrics to score the models. In essence I will use the following:
#
# R^2( Coefficient of determination): This metric measures how well a statistical model predicts the dependent variable. If R^2 test << R^2 train
# , then this indicates that our model does not generalize well to unseen data. (Higher is better)
#
# Root Mean Squared Error: Without using the root or (MSE), it measures the variance of the residuals. The RMSE measures the standard deviation of the errors which occur when a prediction is made on a dataset. They both penalize large prediction errors. (Lower is better)
#
# Mean Absolute Error: MAE measures the average of the absolute difference between the actual and predicted values in the dataset. It is not very sensitive to outliers since it doesn't punish huge errors. (Lower is better)
#
# Mean Absolute Percentage Error: MAPE measures the accuracy of a forecast system. It captures how far off predictions are on average. (Lower is better)

# %%
def get_estimator_scores(models):
    '''
    Uses various metric algorithms to calculate various scores for multiple estimators
    '''
    metrics = []

    for model_name, model in models.items():            
        model.fit(X_train, y_train)
        predictions_test = model.predict(X_test)
        
        metrics.append([
            model_name,
            model.score(X_train, y_train),
            r2_score(y_test, predictions_test),
            np.sqrt(mean_squared_error(y_test, predictions_test)),
            mean_absolute_error(y_test, predictions_test),
            mean_absolute_percentage_error(y_test, predictions_test)
        ])
    
    return pd.DataFrame(metrics, columns=['model', 'r2_train', 'r2_test', 'rmse', 'mae', 'mape'])


# %%
estimator_scores = get_estimator_scores(ensemble_models)


# %%
def plot_estimator_scores(scores):
    melted_r2 = scores[['model', 'r2_train', 'r2_test']].rename(columns={"r2_train": "train", "r2_test": "test"})
    melted_r2 = melted_r2.melt(id_vars='model', var_name='set', value_name='score')
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.4)
    
    sns.barplot(data=melted_r2.round(2), x='score', y='model', hue='set', orient='h', ax=axes[0,0])
    sns.barplot(data=scores.round(2), x='rmse', y='model', orient='h', ax=axes[0,1])
    sns.barplot(data=scores.round(2), x='mae', y='model', orient='h', ax=axes[1,0])
    sns.barplot(data=scores.round(2), x='mape', y='model', orient='h', ax=axes[1,1])
    
    axes[0,0].set_title('R2 Score')
    axes[0,0].bar_label(axes[0,0].containers[0], size=10, padding=5)
    axes[0,0].bar_label(axes[0,0].containers[1], size=10, padding=5)
    axes[0,0].set(xlabel=None, ylabel=None)
    axes[0,0].set_xlim(0, max(melted_r2['score'])+.5)

    axes[0,1].set_title('Root Mean Squared Error')
    axes[0,1].bar_label(axes[0,1].containers[0], size=10, padding=5)
    axes[0,1].set(xlabel=None, ylabel=None)
    axes[0,1].set_xlim(0, max(scores['rmse'])+12)
    
    axes[1,0].set_title('Mean Absolute Error')
    axes[1,0].bar_label(axes[1,0].containers[0], size=10, padding=5)
    axes[1,0].set(xlabel=None, ylabel=None)
    axes[1,0].set_xlim(0, max(scores['mae'])+10)
    
    axes[1,1].set_title('Mean Absolute Percentage Error')
    axes[1,1].bar_label(axes[1,1].containers[0], size=10, padding=5)
    axes[1,1].set(xlabel=None, ylabel=None)
    axes[1,1].set_xlim(0, max(scores['mape'])+0.1)
    
    plt.plot()


# %%
plot_estimator_scores(estimator_scores)

# %% [markdown]
# ### Cross-Validation
# Cross-validation is a technique in machine learning that is used to evaluate predictive performance in estimators. On each iteration, the algorithm splits the input data into two parts, a training set and an evaluation set (folds). The model is then trained on the training fold, and its performance is evaluated against the other validation fold. It is mainly used when we want to estimate how accurately a predictive model will perform and generalize to unseen data.
#
# In this notebook we are dealing with time series data. The dataset contains time records in ascending order and randomly spliting it into various folds will not be ideal, since we want to predict future values. In that case we use another kind of cross-validation called TimeSeriesSplit. This technique splits the time series data into fixed time intervals as train/test sets. These splits **advance in time**, with each new split containing records that must be higher than the previous one.
#
# Let's actually observe the resulting splits for our testing dataset.

# %%
print("Checking for inf values in y_train:", np.isinf(y_train).sum())
print("Checking for NaN values in y_train:", y_train.isna().sum())


# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
tscv = TimeSeriesSplit(n_splits=5)
fig, axes = plt.subplots(tscv.n_splits, 1, figsize=(10, 10), sharex=True)
fig.tight_layout(pad=3.0)


for index, (train_fold, validation_fold) in enumerate(tscv.split(y_train)):
    sns.lineplot(data=y_train.iloc[train_fold], label='Training Set', ax=axes[index])
    sns.lineplot(data=y_train.iloc[validation_fold], label='Validation Set', ax=axes[index])
    axes[index].set_title(f'Time Series Split #{index}')
    axes[index].set(xlabel=None, ylabel=None)
    
plt.show()


# %% [markdown]
# #### Cross-validation for all our models.

# %%
def get_cross_val_scores(models, x, y, cv, scoring):
    '''
    Get cross validated scores for input models.

    Parameters
    ----------
        models (dict): Dictionary containing the name of the model and the estimator object.
        x (DataFrame): A DataFrame containing the feature values to train upon.
        y (DataFrame): A Series object containing the actual predicted values.
        cv (CrossValidator or int): The cross-validation technique. An int value will perform k-fold CV.
        scoring (string): The scoring metric to evaluate the models.

    Return
    ------
        results (DataFrame): A DataFrame which contains the results for the CV run.
    '''
    
    measurements = [(model_name, i, score)
                    for model_name, model in ensemble_models.items()
                    for i, score in enumerate(-cross_val_score(model, x, y, cv=cv, scoring=scoring, n_jobs=N_JOBS))]

    results = pd.DataFrame(measurements, columns=['model', 'fold', 'score'])
    return results


# %%
cv_results = get_cross_val_scores(ensemble_models, X_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error')

# %%
plt.figure(figsize=(8,5))
plt.suptitle('Root Mean Squared Error')

cs_metrics_bxplt = sns.boxplot(x='model', y='score', data=cv_results,whis=[0, 100])
cs_metrics_stplt = sns.stripplot(x='model', y='score', hue='model', data=cv_results,
                                 size=5, jitter=True, linewidth=1, legend=False)

cs_metrics_bxplt.tick_params(labelsize=11)
cs_metrics_bxplt.set(xlabel=None)
plt.tight_layout()
plt.show()

# %%
cv_results.groupby('model').score.mean().sort_values()

# %% [markdown]
# ### Hyperparameter Tuning
# We are now presented with design choices in order to achieve an optimal model architecture. These choices can be made with the form of parameters, which are refered to as **hyperparameters**. Those values are not automatically learned and we have to tune them. However we don't immediatelly know which parameters to tune and we may have to explore a huge range of possibilities. So we create a mapping of hyperparameters and the search space we want to explore.
#
# This is where we can use two popular methods of hyperparameter tuning, the GridSearch and the RandomSearch. The first option exhaustively searches every possible combination of those hyperparaters which yields the best result at the cost of being extremelly slow. The latter option, picks random subsets of the search space thus being faster and usually providing an adequate result. I'm going to use RandomSearch for this notebook.

# %%
# Hyperparameter configurations for RandomizedSearch
model_hyperparameters = {
    'Random Forest': {'n_estimators': [100,150,200],
                      'min_samples_split': [2,5],
                      'min_samples_leaf': [2,4,10],
                      'max_depth': [5,10],
                      'n_jobs': [N_JOBS],
                      'random_state': [RANDOM_STATE]},
    
    'Gradient Boosting': {'learning_rate': np.arange(0.01,1,0.01),
                          'n_estimators': [100,200,300],
                          'min_samples_split': [2,5],
                          'min_samples_leaf': [1,4,10],
                          'max_depth': [3,5],
                          'n_iter_no_change': [10],
                          'tol': [0.01],
                          'random_state': [RANDOM_STATE]},
    
    'AdaBoost': {'learning_rate': np.arange(0.01,1,0.01),
                 'n_estimators': [50,100,200,300],
                 'random_state': [RANDOM_STATE]},
    
    'Histogram GB': {'learning_rate': np.arange(0.01,1,0.01),
                     'max_iter': [100,150,200],
                     'min_samples_leaf': [10,20,30],
                     'max_depth': [None,3,5,10],
                     'n_iter_no_change': [10],
                     'tol': [0.01],
                     'random_state': [RANDOM_STATE]},
    
    'XGBoost': {'learning_rate': np.arange(0.01,1,0.01),
                'n_estimators': [20,50,100,250],
                'max_depth': [None,3,5],
                'eval_metric': ['rmse'],
                'early_stopping_rounds': [10],
                'n_jobs': [N_JOBS],
                'random_state': [RANDOM_STATE]}
}


# %%
def random_search_cv(models, params, n_iter, cv, scoring):
    '''
    Performs hyperparameter tuning using RandomizedSearch.

    Parameters
    ----------
        models (dict): Dictionary containing the name of the model and its respective estimator object.
        params (dict): Dictionary containing the name of the model and its respective hyperparameter spaces to search.
        n_iter (int): The number of candidates to choose from the search space.
        cv (CrossValidator or int): The cross-validation technique. An int value will perform k-fold CV.
        scoring (string): The scoring metric to evaluate the models.

    Return
    ------
        models (dict): A dictionary containing the name of the model and the tuned model parameters.
        model_scores (DataFrame): DataFrame indicating the model's name and the attained best score.
    '''
    
    print(f'Fitting {tscv.n_splits} folds for each of {n_iter} candidates, totalling {tscv.n_splits*n_iter} fits.\n')
    
    model_scores = []

    for model_name, model in ensemble_models.items():
        start = time.time()

        # Use RandomizedSearch as the search space is quite big. For more accurate results we can use GridSearch.
        rscv_model = RandomizedSearchCV(model, params[model_name],
                                        cv=cv,
                                        scoring=scoring,
                                        return_train_score=True,
                                        n_jobs=N_JOBS,
                                        n_iter=n_iter,
                                        random_state=RANDOM_STATE)

        if model_name == 'XGBoost':
            rscv_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=0)
        else:
            rscv_model.fit(X_train, y_train)
        end = time.time()

        print(f'Randomized Search CV for {model_name} finished after {round(end-start, 2)} seconds. Best parameters found:')
        print(f'{rscv_model.best_params_}\n')

        models[model_name] = rscv_model.best_estimator_
        model_scores.append((model_name, round(-rscv_model.best_score_, 4)))
        
    model_scores = pd.DataFrame(model_scores, columns=['model', 'score'])
    
    return models, model_scores


# %%
ensemble_models, rscv_scores = random_search_cv(ensemble_models, model_hyperparameters, n_iter=20, cv=tscv, scoring="neg_root_mean_squared_error")

# %%
fig = plt.figure(figsize=(7,5))
fig.suptitle("RMSE (Train)")

metrics_plt = sns.barplot(rscv_scores.round(2), x='score', y='model', orient='h')
metrics_plt.tick_params(labelsize=10)
metrics_plt.bar_label(metrics_plt.containers[0], size=10, padding=5)

plt.xlim(0, max(rscv_scores.score)+10)
plt.tight_layout()
plt.show()

# %% [markdown]
# Let's now evaluate the tuned models on their ability to predict unseen data (testing set) and also measure the time needed to train and make predictions.

# %%
time_metrics = []
for model_name, model in ensemble_models.items():
    
    fit_start = time.time()
    if model_name == 'XGBoost':
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=0)
    else:
        model.fit(X_train, y_train)
    fit_end = time.time()

    pred_start = time.time()
    predictions_test = model.predict(X_test)
    pred_end = time.time()

    time_metrics.append([
        model_name,
        np.sqrt(mean_squared_error(y_test, predictions_test)),
        fit_end-fit_start,
        pred_end-pred_start
    ])
    
time_metrics = pd.DataFrame(time_metrics, columns=['model', 'rmse', 'fit_time', 'predict_time'])

# %%
fig = plt.figure(figsize=(7,4))
fig.suptitle("RMSE (Test)")
metrics_plt = sns.barplot(time_metrics.round(2), x='rmse', y='model', orient='h')
metrics_plt.tick_params(labelsize=10)
metrics_plt.bar_label(metrics_plt.containers[0], size=10, padding=5)
metrics_plt.set(ylabel=None)

plt.xlim(0, max(time_metrics.rmse)+10)
plt.tight_layout()
plt.show()

# %% [markdown]
# All models perform similarly in the testing set as well. The lowest scores are given by XGBoost and Random Forests, by a tight margin.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
fig.tight_layout(w_pad=2.0)

sns.barplot(time_metrics.round(3), x='fit_time', y='model', orient='h', ax=axes[0])
axes[0].bar_label(axes[0].containers[0], size=10, padding=5)
axes[0].set_xlim(0, max(time_metrics.fit_time)+10)
axes[0].set(xlabel=None, ylabel=None)
axes[0].set_title('Training Time')

sns.barplot(time_metrics.round(3), x='predict_time', y='model', orient='h', ax=axes[1])
axes[1].bar_label(axes[1].containers[0], size=10, padding=5)
axes[1].set_xlim(0, max(time_metrics.predict_time)+0.01)
axes[1].set(xlabel=None, ylabel=None)
axes[1].set_title('Prediction Time')

plt.show()

# %% [markdown]
# Plotting the time needed for training and prediction, we can spot some differences. AdaBoost and Gradient Boosting perform the worse when it comes to model training. In addition to this, Random Forests is moderate in both aspects but not the worst. So in my opinion, if we were to choose one model it would be either the experimental Histogram GB or the pretty famous XGBoost.
#
# #### Feature Importances
# Feature Importance refers to the calculation of the score for all the input features for a given model. These scores represent the importance each feature that was assigned by the model. A higher score means that the specific feature has a higher influence on the model that is used to make predictions.

# %%
feature_importances_df = pd.DataFrame(data=ensemble_models['XGBoost'].feature_importances_,
                                      index=ensemble_models['XGBoost'].feature_names_in_,
                                      columns=['importance'])
feature_importances_df.sort_values('importance').plot(kind='barh', figsize=(6,4)).legend(loc='lower right')


# %% [markdown]
# **Future Predictions**
#
# Next I will let these models make predictions on completely new data about the future (forecasting). We will also visually inspect the results to have a better understanding of how each model tries to come up with future predictions.

# %% [markdown]
# function to create future dataset

# %%
def create_future_dataset_recursive(raw_data, model, target='PM2.5 (ug/m3)', start_year=2025, end_year=2030):
    '''
    Create future dataset and predict recursively until end_year.
    
    Parameters
    ----------
        raw_data (DataFrame): The original dataset with historical AQI data.
        model: The trained XGBoost model.
        target (str): Target column name.
        start_year (int): Starting year for prediction.
        end_year (int): End year for prediction.
        
    Returns
    -------
        future_predictions (DataFrame): DataFrame with predicted AQI values.
    '''
    # Initialize with existing data
    working_data = raw_data.copy()
    
    # Create dataframe to store all predictions
    all_predictions = pd.DataFrame()
    
    # Loop through each year
    for year in range(start_year, end_year + 1):
        print(f"Predicting for year {year}...")
        
        # Create future dates for this year
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31 23:00:00"
        
        # Create the dataset template
        future_df = pd.DataFrame(pd.date_range(start=year_start, end=year_end, freq='1H'), columns=['datetime'])
        future_df = future_df.set_index('datetime')
        future_df = create_features(future_df)  # Your existing feature creation function
        
        # Create lag features using working_data (which contains both historical and previously predicted data)
        for date in future_df.index:
            one_year_ago = date - pd.Timedelta('365 days')
            two_years_ago = date - pd.Timedelta('730 days')
            
            # Check if the dates exist in working_data
            if one_year_ago in working_data.index:
                future_df.loc[date, 'pm_lag_1Y'] = working_data.loc[one_year_ago, target]
            
            if two_years_ago in working_data.index:
                future_df.loc[date, 'pm_lag_2Y'] = working_data.loc[two_years_ago, target]
        
        # Predict AQI for this year
        X_future = future_df[model.feature_names_in_]
        future_df[target] = model.predict(X_future)
        
        # Add predictions to all_predictions
        all_predictions = pd.concat([all_predictions, future_df])
        
        # Update working_data with new predictions for the next iteration
        working_data = pd.concat([working_data, future_df])
    
    return all_predictions



# %%
# Save the XGBoost model
def save_model(model, filename='xgboost_aqi_model.json'):
    """Save the trained XGBoost model to file"""
    model.save_model(filename)
    print(f"Model saved to {filename}")
    
# Example usage
save_model(ensemble_models['XGBoost'])


# %% [markdown]
# ## Visualisations

# %%
def visualize_predictions(predictions_df, city, target='PM2.5 (ug/m3)'):
    """Create visualizations for AQI predictions"""
    # Annual average trend
    yearly_avg = predictions_df.resample('Y').mean()
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_avg.index.year, yearly_avg[target], marker='o')
    plt.title(f'Yearly Average AQI Prediction for {city} (2025-2030)')
    plt.ylabel('AQI')
    plt.xlabel('Year')
    plt.grid(True)
    plt.savefig(f'{city}_yearly_trend.png')
    
    # Monthly patterns for a specific year (e.g., 2030)
    year_2030 = predictions_df[predictions_df.index.year == 2030]
    monthly_2030 = year_2030.resample('M').mean()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_2030.index.month, monthly_2030[target], marker='o')
    plt.title(f'Monthly AQI Pattern for {city} in 2030')
    plt.ylabel('AQI')
    plt.xlabel('Month')
    plt.xticks(range(1,13))
    plt.grid(True)
    plt.savefig(f'{city}_2030_monthly.png')
    
    return yearly_avg, monthly_2030

