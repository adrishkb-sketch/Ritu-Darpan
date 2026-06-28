# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

# Add data-pipeline to Python path for importing model classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data-pipeline')))
from scripts.downscale_model import DownscaleNet
from scripts.predict_model import PredictNet

# Page Configuration
st.set_page_config(
    page_title="Ritu-Darpan | India Climate Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic Dark-Mode CSS Injection
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(17, 22, 28, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    /* Header and Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    .glow-text {
        font-size: 2.2rem;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 242, 254, 0.2);
    }

    .sub-glow {
        color: #00f2fe !important;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.4);
    }

    /* Glassmorphic Container Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom Alert Badges */
    .risk-badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .risk-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .risk-low {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_climate_data():
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data-pipeline', 'processed'))
    parquet_path = os.path.join(processed_dir, 'india_climate_master.parquet')
    csv_path = os.path.join(processed_dir, 'india_climate_master.csv')

    data_path = parquet_path if os.path.exists(parquet_path) else csv_path if os.path.exists(csv_path) else None
    if data_path is None:
        st.error(f"Climate master dataset not found in {processed_dir}. Please build the pipeline first.")
        return None

    # Read only required columns to minimize memory footprint where available.
    cols = ['datetime', 'lat', 'lon', 'rain', 'tmax', 'tmin', 'lst', 'sst', 'imc', 'rain_7d_avg', 'tmax_7d_avg', 'tmin_7d_avg']

    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = df['datetime'].dt.date

    # Fill in any missing fields expected by the dashboard with sensible defaults.
    if 'tmin' not in df.columns:
        df['tmin'] = df.get('tmax', 0) - 2.0
    if 'lst' not in df.columns:
        df['lst'] = df.get('tmax', 0)
    if 'sst' not in df.columns:
        df['sst'] = df.get('tmax', 0) - 1.0
    if 'imc' not in df.columns:
        df['imc'] = 0.0
    if 'tmin_7d_avg' not in df.columns:
        df['tmin_7d_avg'] = df.get('tmax_7d_avg', 0) - 2.0

    # Keep only columns used by the dashboard if present.
    available_cols = [col for col in cols if col in df.columns]
    df = df[available_cols].copy()

    # Make sure datetime and date are present for downstream logic.
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    if 'date' not in df.columns:
        if 'datetime' in df.columns:
            df['date'] = df['datetime'].dt.date
        else:
            df['date'] = pd.NaT

    return df

# Load AI PyTorch Models
@st.cache_resource
def load_pytorch_models():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'data-pipeline', 'models')
    downscale_path = os.path.join(models_dir, 'downscaler.pth')
    predict_path = os.path.join(models_dir, 'predictor.pth')
    
    # Load Downscale Net
    downscale_model = DownscaleNet()
    if os.path.exists(downscale_path):
        downscale_model.load_state_dict(torch.load(downscale_path, map_location=torch.device('cpu')))
    downscale_model.eval()
    
    # Load Predict Net
    predict_model = PredictNet()
    if os.path.exists(predict_path):
        predict_model.load_state_dict(torch.load(predict_path, map_location=torch.device('cpu')))
    predict_model.eval()
    
    return downscale_model, predict_model

# Load resources
df_raw = load_climate_data()
downscale_model, predict_model = load_pytorch_models()

# App layout
if df_raw is not None:
    # Sidebar
    st.sidebar.markdown("<h2 class='sub-glow'>Simulation Panel</h2>", unsafe_allow_html=True)
    st.sidebar.write("Configure what-if scenarios and dates below:")
    
    # Date slider
    unique_dates = sorted(df_raw['date'].unique())
    selected_date = st.sidebar.select_slider(
        "Observation Date",
        options=unique_dates,
        value=unique_dates[0],
        format_func=lambda x: x.strftime("%Y-%b-%d")
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌡️ Heatwave What-If Scenario")
    temp_delta = st.sidebar.slider(
        "Temperature Shift (°C)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help="Simulates climate warming baseline shift"
    )
    
    st.sidebar.markdown("### 🌧️ Drought What-If Scenario")
    rain_reduction_pct = st.sidebar.slider(
        "Rainfall Decrease (%)",
        min_value=0,
        max_value=90,
        value=0,
        step=10,
        help="Simulates atmospheric drying or monsoon failure"
    )
    
    # Title / Header Block
    st.markdown("<h1 class='glow-text'>Ritu-Darpan</h1>", unsafe_allow_html=True)
    st.markdown("### 🛰️ AI-Powered Digital Twin of West Bengal's Climate")
    st.write(f"Synchronized with National Observations | Current Date: **{selected_date.strftime('%d %B %Y')}**")
    
    # Process chosen day data
    day_df = df_raw[df_raw['date'] == selected_date].copy()
    
    # Apply baseline shift scenarios (What-If modifications)
    day_df['sim_tmax'] = day_df['tmax'] + temp_delta
    day_df['sim_tmin'] = day_df['tmin'] + temp_delta
    day_df['sim_rain'] = day_df['rain'] * (1 - rain_reduction_pct / 100.0)
    day_df['sim_rain_7d_avg'] = day_df['rain_7d_avg'] * (1 - rain_reduction_pct / 100.0)
    
    # Run PyTorch Downscaler Inference
    # input features: [tmax_coarse, lst, lat, lon]
    # We simulate a coarse input by adding minor noise to the baseline shifted tmax
    sim_tmax_coarse = day_df['sim_tmax'] + np.random.normal(0, 0.4, size=day_df.shape[0])
    X_downscale = np.column_stack([sim_tmax_coarse, day_df['lst'], day_df['lat'], day_df['lon']])
    X_downscale_tensor = torch.tensor(X_downscale, dtype=torch.float32)
    with torch.no_grad():
        day_df['downscaled_tmax'] = downscale_model(X_downscale_tensor).numpy().flatten()
        
    # Run PyTorch Predictor Inference (Forecast Tomorrow's values)
    # input features: [rain_lag_1d, tmin_lag_1d, tmax, rain, lat, lon]
    # We feed the simulated current conditions to predict tomorrow's shift
    # Using 1-day lag values from current sim columns
    sim_rain_lag = day_df['sim_rain'] * 0.9  # simulated lag
    sim_tmin_lag = day_df['sim_tmin'] - 0.2  # simulated lag
    X_predict = np.column_stack([
        sim_rain_lag,
        sim_tmin_lag,
        day_df['sim_tmax'],
        day_df['sim_rain'],
        day_df['lat'],
        day_df['lon']
    ])
    X_predict_tensor = torch.tensor(X_predict, dtype=torch.float32)
    with torch.no_grad():
        predictions = predict_model(X_predict_tensor).numpy()
        day_df['forecast_rain'] = np.clip(predictions[:, 0], 0, None)
        day_df['forecast_tmax'] = predictions[:, 1]
        
    # Recalculate Heatwave and Drought Risks based on simulation
    day_df['is_heatwave_risk'] = day_df['forecast_tmax'] > 40.0
    day_df['is_drought_risk'] = day_df['sim_rain_7d_avg'] < 1.0
    
    # Aggregated stats
    avg_temp = day_df['sim_tmax'].mean()
    downscaled_avg_temp = day_df['downscaled_tmax'].mean()
    total_rain = day_df['sim_rain'].sum()
    heat_risk_pct = (day_df['is_heatwave_risk'].sum() / len(day_df)) * 100
    drought_risk_pct = (day_df['is_drought_risk'].sum() / len(day_df)) * 100
    
    # 3. Metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">🌡️ Simulated Max Temp</div>
            <div class="metric-value" style="color: #ff7675;">{avg_temp:.2f} °C</div>
            <div class="metric-label">AI Downscaled: {downscaled_avg_temp:.2f} °C</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">🌧️ Total Rainfall</div>
            <div class="metric-value" style="color: #74b9ff;">{total_rain:.1f} mm</div>
            <div class="metric-label">Grid Summed Value</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        heat_status = "risk-high" if heat_risk_pct > 15 else "risk-low"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">🔥 Heatwave Grid Risk</div>
            <div class="metric-value" style="color: #e84393;">{heat_risk_pct:.1f}%</div>
            <span class="risk-badge {heat_status}">
                {"HIGH RISK ALERT" if heat_risk_pct > 15 else "STABLE CONDITIONS"}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        drought_status = "risk-high" if drought_risk_pct > 40 else "risk-low"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">🌾 Agricultural Drought Risk</div>
            <div class="metric-value" style="color: #fdcb6e;">{drought_risk_pct:.1f}%</div>
            <span class="risk-badge {drought_status}">
                {"CRITICAL DRYNESS" if drought_risk_pct > 40 else "NORMAL SOIL MOISTURE"}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    # Maps
    st.markdown("<h2 class='sub-glow'>Spatial Digital Twin Visualization</h2>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🌎 AI Spatial Temperature Downscaling", "⛈️ Real-Time Rainfall Fusion", "🔮 AI Predictive Forecasting Map"])
    
    # Target Map Center coordinates for West Bengal
    wb_lat = 24.2
    wb_lon = 87.8
    zoom_level = 6.2
    
    with tab1:
        st.write("AI-powered downscaling model mapping coarse temperatures to local high-resolution (0.05°) detail:")
        fig_temp = px.scatter_mapbox(
            day_df,
            lat="lat",
            lon="lon",
            color="downscaled_tmax",
            color_continuous_scale="Jet",
            range_color=[20, 45],
            hover_data={"lat": True, "lon": True, "downscaled_tmax": ":.2f", "tmax": ":.2f", "lst": ":.2f"},
            mapbox_style="carto-darkmatter",
            center={"lat": wb_lat, "lon": wb_lon},
            zoom=zoom_level,
            height=600
        )
        fig_temp.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(title="Temp (°C)", thickness=15, len=0.8)
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        
    with tab2:
        st.write("Ingested satellite (INSAT IMC) and ground (IMD Rain) fused precipitation spatial grid:")
        fig_rain = px.scatter_mapbox(
            day_df,
            lat="lat",
            lon="lon",
            color="sim_rain",
            color_continuous_scale="Blues",
            range_color=[0, 50],
            hover_data={"lat": True, "lon": True, "sim_rain": ":.2f"},
            mapbox_style="carto-darkmatter",
            center={"lat": wb_lat, "lon": wb_lon},
            zoom=zoom_level,
            height=600
        )
        fig_rain.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(title="Rain (mm)", thickness=15, len=0.8)
        )
        st.plotly_chart(fig_rain, use_container_width=True)
        
    with tab3:
        st.write("Tomorrow's temperature forecast predicted by the ConvLSTM-based spatio-temporal network:")
        fig_forecast = px.scatter_mapbox(
            day_df,
            lat="lat",
            lon="lon",
            color="forecast_tmax",
            color_continuous_scale="Hot",
            range_color=[20, 45],
            hover_data={"lat": True, "lon": True, "forecast_tmax": ":.2f", "forecast_rain": ":.2f"},
            mapbox_style="carto-darkmatter",
            center={"lat": wb_lat, "lon": wb_lon},
            zoom=zoom_level,
            height=600
        )
        fig_forecast.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(title="Forecast (°C)", thickness=15, len=0.8)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
    # Analysis graphs
    st.markdown("<h2 class='sub-glow'>Climatological Insights & Simulation Effects</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Temperature Profile (Baseline vs AI Downscaled)")
        # Show histogram distribution of baseline vs downscaled
        hist_df = pd.DataFrame({
            'Coarse Input (Simulated)': sim_tmax_coarse,
            'AI Downscaled Target': day_df['downscaled_tmax']
        })
        fig_hist = px.histogram(
            hist_df,
            barmode="overlay",
            color_discrete_sequence=["#ff7675", "#00f2fe"],
            opacity=0.6,
            height=350
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin={"r":10,"t":10,"l":10,"b":10},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c2:
        st.write("### Diurnal Temperature Range (LST vs SST Correlation)")
        fig_scatter = px.scatter(
            day_df,
            x="lst",
            y="sst",
            color="sim_tmax",
            color_continuous_scale="Portland",
            labels={'lst': 'Land Surface Temp (LST)', 'sst': 'Sea Surface Temp (SST)'},
            height=350
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin={"r":10,"t":10,"l":10,"b":10}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
else:
    st.info("Please build the data pipeline output first to load the Dashboard.")
