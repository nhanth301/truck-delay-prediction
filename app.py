import pandas as pd
import streamlit as st
import hopsworks
from pipeline.utils import load_config
from workflow.nodes.check_model_drift import init_model

st.set_page_config(
    page_title="Truck Delay Prediction",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
        color: #2c3e50;
    }

    /* Header */
    .main-header {
        background: #2c3e50;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .main-subtitle {
        color: #dee2e6;
        font-size: 1.1rem;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
    }

    /* Alerts */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border-left: 6px solid #28a745 !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stWarning {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border-left: 6px solid #ffc107 !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stError {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border-left: 6px solid #dc3545 !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Buttons */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #2c3e50, #4c6ef5);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚛 Truck Delay Prediction</h1>
        <p class="main-subtitle">AI-powered logistics optimization with real-time delay predictions</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_conf():
    return load_config()

@st.cache_resource
def load_model():
    return init_model(config)

config = load_conf()
hopsworks_key = config["hopsworks"]["api_key"]

with st.spinner("🔄 Loading ML models and configurations..."):
    model, scaler, encoder = load_model()
    cts_cols = config["features"]["cts_col_names"]
    cat_cols = config["features"]["cat_col_names"]
    encode_columns = config["features"]["encode_column_names"]

@st.cache_data
def load_feature_data():
    try:
        project = hopsworks.login(api_key_value=hopsworks_key)
        fs = project.get_feature_store()
        fg = fs.get_feature_group("final_data", version=1)
        query = fg.select_all()
        df = query.read()
        return df.dropna() if not df.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error fetching data from Hopsworks: {e}")
        return pd.DataFrame()

final_merge = load_feature_data()
if final_merge.empty:
    st.error("❌ No data available. Please check your Hopsworks connection.")
    st.stop()

# ==============================
# Filters
# ==============================
st.sidebar.header("🔍 Filters")

use_date_filter = st.sidebar.checkbox("📅 Date Range")
if use_date_filter:
    from_date = st.sidebar.date_input("Start", value=min(final_merge["departure_date"]))
    to_date = st.sidebar.date_input("End", value=max(final_merge["departure_date"]))

use_truck_filter = st.sidebar.checkbox("🚚 Truck ID")
if use_truck_filter:
    truck_id = st.sidebar.selectbox("Truck ID:", sorted(final_merge["truck_id"].unique()))

use_route_filter = st.sidebar.checkbox("🛣️ Route ID")
if use_route_filter:
    route_id = st.sidebar.selectbox("Route ID:", sorted(final_merge["route_id"].unique()))

st.sidebar.markdown("---")
st.sidebar.metric("Total Records", f"{len(final_merge):,}")
st.sidebar.metric("Unique Trucks", final_merge["truck_id"].nunique())
st.sidebar.metric("Unique Routes", final_merge["route_id"].nunique())


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="stat-number">{len(final_merge):,}</div>
            <div class="stat-label">Total Shipments</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    delay_rate = (final_merge["delay"].sum() / len(final_merge) * 100) if len(final_merge) > 0 else 0
    st.markdown(f"""
        <div class="metric-card">
            <div class="stat-number">{delay_rate:.1f}%</div>
            <div class="stat-label">Delay Rate</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="stat-number">{final_merge["truck_id"].nunique()}</div>
            <div class="stat-label">Active Trucks</div>
        </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="stat-number">{final_merge["route_id"].nunique()}</div>
            <div class="stat-label">Route Network</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 🎯 Run Prediction Analysis")
if st.button("🚀 Generate Predictions"):
    with st.spinner("🔮 Running ML predictions..."):
        try:
            data = final_merge.copy()
            if use_date_filter:
                data = data[(data["departure_date"] >= str(from_date)) & (data["departure_date"] <= str(to_date))]
            if use_truck_filter:
                data = data[data["truck_id"] == truck_id]
            if use_route_filter:
                data = data[data["route_id"] == str(route_id)]

            if data.empty:
                st.warning("⚠️ No data found for the selected filters.")
                st.stop()

            # Preprocess
            X_test = data[cts_cols + cat_cols]
            encoded_features = list(encoder.get_feature_names_out(encode_columns))
            X_test[encoded_features] = encoder.transform(X_test[encode_columns])
            X_test[cts_cols] = scaler.transform(X_test[cts_cols])
            X_test = X_test.drop(encode_columns, axis=1)

            # Predict
            y_preds = model.predict(X_test)
            result = pd.DataFrame({
                "truck_id": data["truck_id"].values,
                "route_id": data["route_id"].values,
                "departure_date": data["departure_date"].values,
                "estimated_arrival": data["estimated_arrival"].values,
            })
            result["prediction"] = y_preds
            result["prediction"] = result["prediction"].apply(lambda x: "🔴 Delayed" if x == 1 else "🟢 On Time")

            if not result.empty:
                st.markdown("### 📋 Detailed Predictions")
                st.dataframe(result, use_container_width=True, hide_index=True)

                csv = result.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name=f"truck_delay_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("⚠️ Model returned no predictions.")
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 1rem;'>
        🚛 <strong>Truck Delay Prediction Dashboard</strong> | Clean UI with Machine Learning
    </div>
""", unsafe_allow_html=True)
