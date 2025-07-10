import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import sqlite3
from aqi_functions import create_future_dataset_direct, get_health_risk,create_monthly_summary, COLOR_SCHEME
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
# icon = Image.open("assets/logo1_bg.PNG")
st.set_page_config(
    page_title="India AQI Forecast", 
    layout="wide",
    page_icon="🌫️"
    # page_icon=icon
    
)

STATES_DIR = "data/processed_states"
MODELS_DIR = "models/state_models"
DB_PATH = "precomputed_forecasts.db"


st.markdown(f"""
<style>
    .stApp {{
        background-color: #ffffff  !important;
        font-family:'Poppins', sans-serif;
    }}
    .css-1d391kg {{
        background-color: {COLOR_SCHEME['navy']} !important;
        color: white !important;
    }}
    .stButton>button {{
        background-color: {COLOR_SCHEME['azul']} !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
    }}
    .stMetric {{
        background-color: {COLOR_SCHEME['light']} !important;
        border-radius: 10px !important;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }}
    .stProgress > div > div {{
        background-color: {COLOR_SCHEME['yellow']} !important;
    }}
    div[data-baseweb="select"] > div {{
    cursor: pointer !important;
    }}
    /* Hide link icons on headers */
    [data-testid="stMarkdownContainer"] a {{
        display: none;
    }}
    
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_state_model(state):
    model = xgb.XGBRegressor()
    model.load_model(f"{MODELS_DIR}/xgboost_{state}_model.json")
    
    with open(f"{MODELS_DIR}/{state}_features.txt", "r") as f:
        features = f.read().splitlines()
    
    if 'PM2.5 (ug/m3)' in features:
        target = 'PM2.5 (ug/m3)'
    elif 'PM10 (ug/m3)' in features:
        target = 'PM10 (ug/m3)'
    else:
        target = features[-1]
    return model, features, target

@st.cache_data
def load_state_data(state):
    df= pd.read_csv(f"{STATES_DIR}/{state}_processed.csv", 
                      parse_dates=['datetime'], 
                      index_col='datetime')
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep='first')]
    return df

@st.cache_data(ttl=3600)
def get_precomputed_forecast(state, year):
    """Retrieve precomputed forecast with error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = '''
            SELECT timestamp, prediction 
            FROM forecasts 
            WHERE state = ? AND year = ?
            ORDER BY timestamp
        '''
        df = pd.read_sql_query(query, conn, params=(state, year))
        conn.close()
        
        if df.empty:
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df['prediction']
    
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            st.warning("Precomputed forecasts not available. Using live prediction.")
            return None
        raise


def create_aqi_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        number = {'font': {'size': 28, 'color': COLOR_SCHEME['navy']}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18, 'color': COLOR_SCHEME['navy'],'family': 'Poppins, sans-serif'}},
        gauge = {
            'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': COLOR_SCHEME['navy']},
            'bar': {'color': COLOR_SCHEME['navy'],'thickness': 0.2},
            'steps': [
                {'range': [0, 50], 'color': COLOR_SCHEME['green']},
                {'range': [50, 100], 'color': COLOR_SCHEME['light-green']},
                {'range': [100, 200], 'color': COLOR_SCHEME['yellow']},
                {'range': [200, 300], 'color': COLOR_SCHEME['orange']},
                {'range': [300, 400], 'color': COLOR_SCHEME['red']},
                {'range': [400, 500], 'color': COLOR_SCHEME['dark-red']}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 1},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=10),
        height=200,
        width=400
    )
    return fig


def main():    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500&display=swap');

        .custom-header {{
            font-family: 'Poppins', sans-serif;
            color: {COLOR_SCHEME['navy']};
            text-align: center;
            margin-top: -80px;
        }}
    </style>
    <div style="padding:20px;border-radius:10px">
        <h1 class="custom-header">National Air Quality Prediction System</h1>
    </div>
""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div style="font-family: Poppins, sans-serif;">', unsafe_allow_html=True)
        st.markdown("### Select State")
        raw_states = [f.replace('_processed.csv', '') for f in os.listdir(STATES_DIR)
                      if f.endswith('_processed.csv') and f.replace('_processed.csv', '') not in ['Arunachal_Pradesh', 'Jharkhand']]

        state_display_map = {s.replace('_', ' '): s for s in raw_states}
        state = state_display_map[st.selectbox("", list(state_display_map.keys()))]
        
        st.markdown("### Prediction Year")
        year = st.selectbox("", list(range(2024, 2026)))

        
        if st.button("Generate Forecast", key="predict_btn"):
            st.session_state.predict = True

    with col2:
        st.markdown('<div style="font-family: Poppins, sans-serif;">', unsafe_allow_html=True)
        if 'predict' in st.session_state and st.session_state.predict:
            with st.spinner("Loading data and generating predictions..."):
                try:
                    predictions = get_precomputed_forecast(state, year)
                    model, features, target = load_state_model(state)
                    if predictions is None:
                        #fallback
                        df = load_state_data(state)
                        

                        future_df = create_future_dataset_direct(
                            raw_data=df,
                            target=target,
                            start_date=f"{year}-01-01",
                            end_date=f"{year}-12-31 23:00:00",
                            required_features=features
                        )
                        predictions = model.predict(future_df)
                        future_df[target] = predictions
                    else:
                        future_df = pd.DataFrame({
                            target: predictions
                        }, index=pd.date_range(
                            start=f"{year}-01-01", 
                            end=f"{year}-12-31 23:00:00",
                            freq='h'
                        ))

                    
                    avg_aqi = predictions.mean()
                    risk = get_health_risk(avg_aqi) #can use if needed
                    
                    
                    future_series = pd.Series(predictions, index=future_df.index)
                    

                    fig, ax = plt.subplots(figsize=(14, 4))
                    
                    sns.lineplot(x=future_series.index, y=future_series.values, 
                                 label=f"{year} Forecast", color=COLOR_SCHEME['polynesian_blue'], ax=ax)

                    ax.set_title(f"{state} - {target} Forecast")
                    ax.set_ylabel(target)
                    ax.set_xlabel("Date")
                    ax.legend()
                    st.pyplot(fig)
                    
                    
                    with st.expander("AVG AQI Prediction", expanded=False):
                        st.plotly_chart(
                            create_aqi_gauge(avg_aqi, "Average AQI Level"), 
                            use_container_width=True
                        )
                    monthly_summary = create_monthly_summary(pd.Series(predictions, index=future_df.index))

                    st.markdown(
                        f"""
                        <h3 style="font-family: 'Poppins', sans-serif; font-weight: 600; color: #222;">
                            Monthly AQI Summary for {year}
                        </h3>
                        """,
                        unsafe_allow_html=True
                    )
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.markdown("**Month**")
                    with col2:
                        st.markdown("**Mean AQI**")
                    with col3:
                        st.markdown("**Risk level**")
                    
                    for _, row in monthly_summary.iterrows():
                        cols = st.columns([3, 2, 1])
                        with cols[0]:
                            st.write(row['Month'])
                        with cols[1]:
                            st.write(row['Mean AQI'])
                        with cols[2]:
                            st.markdown(
                                f'<div style="background-color:{row["Color"]}; width:20px; height:20px; border-radius:3px;"></div>',
                                unsafe_allow_html=True
                            )
                    
                    #Health risk legend
                    st.markdown(
                        '<h4 style="font-size:18px; font-family:Poppins, sans-serif; color:#333;">Legend</h4>',
                        unsafe_allow_html=True
                    )
                    risk_categories = [
                        ("Good (0-50)",COLOR_SCHEME["green"]),
                        ("Satisfactory (51-100)", COLOR_SCHEME["light-green"]),
                        ("Moderate (101-200)", COLOR_SCHEME["yellow"]),
                        ("Poor (201-300)", COLOR_SCHEME['orange']),
                        ("Very Poor (301-400)", COLOR_SCHEME['red']),
                        ("Severe (401-500)", COLOR_SCHEME['dark-red'])
                    ]
                    
                    for category, color in risk_categories:
                        st.markdown(
                            f'''
                            <div style="display:flex; align-items:center; margin-bottom:5px;">
                                <div style="background-color:{color}; width:8px; height:8px; border-radius:2px; margin-right:10px;"></div>
                                <span style= font-size:14px; color:#222;">{category}</span>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
                                       
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
    #disclaimer 
    st.markdown(
        """
        <hr style="margin-top:80px; margin-bottom:10px">
        <div style="font-size:14px; color:gray; line-height:1.6;">
            <b>Disclaimer:</b> The states <span style="color:#c41e3d;"><b>Arunachal Pradesh</b></span> and 
            <span style="color:#c41e3d;"><b>Jharkhand</b></span> have a large amount of missing and insufficient data, and 
            are therefore excluded from predictions. <br>
            This application provides estimated AQI forecasts using a machine learning model trained on historical data. 
            For official and real-time AQI values, please refer to authoritative government sources or state pollution control boards.<br>
            
        </div>
        
        """,
        unsafe_allow_html=True
    )
    

if __name__ == "__main__":
    main()
