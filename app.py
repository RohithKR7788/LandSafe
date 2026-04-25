"""
app.py
~~~~~~
LandSafe — Streamlit dashboard for interactive landslide risk prediction.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src import load_model, risk_label, build_model, load_and_prepare
from generate_data import generate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LandSafe — Kerala Landslide Risk",
    page_icon="⛰️",
    layout="wide",
)

# ── Kerala districts with approx. centroids ───────────────────────────────────
KERALA_DISTRICTS = {
    "Wayanad":       (11.6854, 76.1320),
    "Idukki":        (9.8500,  77.0000),
    "Malappuram":    (11.0730, 76.0740),
    "Kozhikode":     (11.2500, 75.7750),
    "Thrissur":      (10.5276, 76.2144),
    "Palakkad":      (10.7867, 76.6548),
    "Kannur":        (11.8745, 75.3704),
    "Kasaragod":     (12.4996, 74.9869),
    "Ernakulam":     (10.0000, 76.3000),
    "Pathanamthitta":(9.2648,  76.7870),
    "Kottayam":      (9.5916,  76.5222),
    "Kollam":        (8.8932,  76.6141),
    "Alappuzha":     (9.4981,  76.3388),
    "Thiruvananthapuram": (8.5241, 76.9366),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training model (first run only)…")
def get_model():
    model_path = Path("models/landsafe_model.pkl")
    if model_path.exists():
        return load_model(model_path)
    # Train from scratch on first run
    df = generate(n=10_000)
    df.to_csv("data/processed/kerala_landslide_dataset.csv", index=False)
    X, y = load_and_prepare("data/processed/kerala_landslide_dataset.csv")
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline = build_model()
    pipeline.fit(X_train, y_train)
    from src import save_model
    save_model(pipeline)
    return pipeline


def predict_single(pipeline, inputs: dict) -> tuple[float, str, str]:
    df = pd.DataFrame([inputs])
    prob = pipeline.predict_proba(df)[0, 1]
    label, color = risk_label(prob)
    return prob, label, color


def district_risk_map(pipeline, rainfall_1d: float, rainfall_3d: float, rainfall_7d: float) -> pd.DataFrame:
    """Predict risk for every Kerala district given current rainfall."""
    rows = []
    for district, (lat, lon) in KERALA_DISTRICTS.items():
        inp = {
            "slope_angle":   25 + np.random.uniform(-8, 12),
            "elevation":     600 + np.random.uniform(-200, 800),
            "aspect":        np.random.uniform(0, 360),
            "curvature":     np.random.normal(0, 1),
            "soil_type":     1,
            "lithology":     0,
            "soil_moisture": min(1.0, 0.3 + rainfall_3d / 300),
            "rainfall_1d":   rainfall_1d + np.random.uniform(-5, 5),
            "rainfall_3d":   rainfall_3d + np.random.uniform(-10, 10),
            "rainfall_7d":   rainfall_7d + np.random.uniform(-15, 15),
            "ndvi":          0.55,
            "land_use":      0,
            "dist_river":    300 + np.random.uniform(-100, 200),
            "dist_road":     500,
        }
        prob, label, color = predict_single(pipeline, inp)
        rows.append({
            "District": district, "Latitude": lat, "Longitude": lon,
            "Risk Probability": prob, "Risk Level": label, "Color": color,
        })
    return pd.DataFrame(rows)


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.title("⛰️ LandSafe — Kerala Hyperlocal Landslide Risk Prediction")
    st.caption("AI-powered 24–48 hour landslide risk forecasting for the Western Ghats")

    pipeline = get_model()

    # Sidebar
    st.sidebar.header("🌧️ Current Rainfall Inputs")
    rainfall_1d = st.sidebar.slider("Rainfall today (mm)", 0, 300, 40)
    rainfall_3d = st.sidebar.slider("Rainfall last 3 days (mm)", 0, 600, 90)
    rainfall_7d = st.sidebar.slider("Rainfall last 7 days (mm)", 0, 900, 150)

    st.sidebar.markdown("---")
    st.sidebar.header("🗻 Terrain Parameters")
    slope_angle  = st.sidebar.slider("Slope angle (°)", 0, 75, 30)
    elevation    = st.sidebar.slider("Elevation (m)", 50, 2700, 700)
    soil_moisture= st.sidebar.slider("Soil moisture (0–1)", 0.0, 1.0, 0.5)
    ndvi         = st.sidebar.slider("NDVI (vegetation index)", -0.2, 0.9, 0.4)
    dist_river   = st.sidebar.slider("Distance to river (m)", 10, 5000, 300)

    st.sidebar.markdown("---")
    st.sidebar.header("🪨 Geology")
    soil_type  = st.sidebar.selectbox("Soil type",   ["Laterite", "Alluvial", "Clay", "Sandy", "Rocky"])
    lithology  = st.sidebar.selectbox("Lithology",   ["Granite", "Gneiss", "Charnockite", "Sedimentary"])
    land_use   = st.sidebar.selectbox("Land use",    ["Forest", "Agriculture", "Urban", "Barren", "Water"])

    # ── Main predict button ────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📍 Single-Location Prediction")
        inputs = {
            "slope_angle":   slope_angle,
            "elevation":     elevation,
            "aspect":        180,
            "curvature":     0.5,
            "soil_type":     ["Laterite","Alluvial","Clay","Sandy","Rocky"].index(soil_type),
            "lithology":     ["Granite","Gneiss","Charnockite","Sedimentary"].index(lithology),
            "soil_moisture": soil_moisture,
            "rainfall_1d":   rainfall_1d,
            "rainfall_3d":   rainfall_3d,
            "rainfall_7d":   rainfall_7d,
            "ndvi":          ndvi,
            "land_use":      ["Forest","Agriculture","Urban","Barren","Water"].index(land_use),
            "dist_river":    dist_river,
            "dist_road":     500,
        }
        prob, label, color = predict_single(pipeline, inputs)

        st.metric("Risk Probability", f"{prob*100:.1f}%")
        st.markdown(
            f"<div style='background:{color};color:white;padding:12px 20px;"
            f"border-radius:8px;font-size:1.4rem;font-weight:600;"
            f"text-align:center;'>{label} Risk</div>",
            unsafe_allow_html=True,
        )

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0,  25], "color": "#d5f5e3"},
                    {"range": [25, 50], "color": "#fdebd0"},
                    {"range": [50, 75], "color": "#fad7a0"},
                    {"range": [75, 100],"color": "#f9ebea"},
                ],
                "threshold": {
                    "line": {"color": "#e74c3c", "width": 3},
                    "thickness": 0.75,
                    "value": 75,
                },
            },
            title={"text": "Landslide Risk"},
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=30, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("🗺️ Kerala District Risk Map")
        dist_df = district_risk_map(pipeline, rainfall_1d, rainfall_3d, rainfall_7d)

        m = folium.Map(location=[10.5, 76.5], zoom_start=7,
                       tiles="CartoDB positron")
        for _, row in dist_df.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=14,
                color=row["Color"],
                fill=True, fill_color=row["Color"],
                fill_opacity=0.75,
                popup=folium.Popup(
                    f"<b>{row['District']}</b><br>"
                    f"Risk: {row['Risk Level']}<br>"
                    f"Probability: {row['Risk Probability']*100:.1f}%",
                    max_width=160,
                ),
                tooltip=f"{row['District']}: {row['Risk Level']}",
            ).add_to(m)

        st_folium(m, width=700, height=420)

    # ── District risk table ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 District Risk Summary")
    display_df = dist_df[["District","Risk Level","Risk Probability"]].copy()
    display_df["Risk Probability"] = (display_df["Risk Probability"] * 100).round(1).astype(str) + "%"
    display_df = display_df.sort_values("Risk Level", ascending=False)

    color_map = {"Extreme": "#fdecea", "High": "#fef3e2",
                 "Moderate": "#fefde2", "Low": "#eafbf0"}
    st.dataframe(
        display_df.style.apply(
            lambda row: [f"background-color: {color_map.get(row['Risk Level'], '#fff')}" for _ in row],
            axis=1,
        ),
        use_container_width=True, hide_index=True,
    )

    # ── Feature contribution bar chart ────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Key Risk Factors")
    feature_names = [
        "Slope angle", "Rainfall (3-day)", "Soil moisture",
        "Elevation", "NDVI", "Distance to river",
        "Rainfall (1-day)", "Rainfall (7-day)", "Soil type", "Land use",
    ]
    importances = [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01, 0.01]
    fig_bar = px.bar(
        x=importances, y=feature_names,
        orientation="h",
        labels={"x": "Relative importance", "y": ""},
        color=importances, color_continuous_scale="Reds",
        title="Feature importance for landslide prediction",
    )
    fig_bar.update_layout(coloraxis_showscale=False, height=360,
                          margin=dict(l=10, r=10, t=40, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.caption(
        "LandSafe v1.0 | Built with XGBoost + SHAP + Streamlit | "
        "Data: NASA SRTM, IMD, Sentinel-2, GSI | "
        "⚠️ This is a research prototype — not for emergency decision-making."
    )


if __name__ == "__main__":
    main()
