import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="AURAVERDE AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
# 🌱 AURAVERDE Intelligent Aquaponics Platform

### AI + IoT + Sustainability Analytics
""")

data = pd.read_csv(
    "auraverde_water_quality_dataset.csv"
)

# ---------------------------
# CREATE SYSTEM HEALTH
# ---------------------------

data["system_health"] = (
    0.3 * data["pH"] +
    0.4 * data["dissolved_oxygen_mg_L"] -
    0.3 * data["ammonia_mg_L"]
)

# ---------------------------
# MODEL TRAINING
# ---------------------------

X = data.drop(
    columns=["timestamp", "system_health"]
)

y = data["system_health"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestRegressor()

model.fit(
    X_train_scaled,
    y_train
)

# ---------------------------
# SIDEBAR INPUTS
# ---------------------------

st.sidebar.header("Input Parameters")

pH = st.sidebar.slider(
    "pH",
    4.0,
    10.0,
    7.0
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    15.0,
    40.0,
    28.0
)

do = st.sidebar.slider(
    "Dissolved Oxygen",
    1.0,
    10.0,
    6.0
)

turbidity = st.sidebar.slider(
    "Turbidity",
    0.0,
    10.0,
    3.0
)

ammonia = st.sidebar.slider(
    "Ammonia",
    0.0,
    3.0,
    0.5
)

# ---------------------------
# CREATE INPUT DATAFRAME
# ---------------------------

input_data = pd.DataFrame({
    "pH": [pH],
    "temperature_C": [temperature],
    "dissolved_oxygen_mg_L": [do],
    "turbidity_NTU": [turbidity],
    "ammonia_mg_L": [ammonia]
})

# ---------------------------
# PREDICTION
# ---------------------------

scaled_input = scaler.transform(input_data)

prediction = model.predict(scaled_input)[0]

# ---------------------------
# SUSTAINABILITY SCORE
# ---------------------------

water_quality = (
    pH + do
) / 2

score = (
    0.7 * water_quality
    -
    0.3 * ammonia
)

sustainability_score = max(
    0,
    min(100, score * 10)
)

# ---------------------------
# AI RECOMMENDATION
# ---------------------------

recommendations = []

if do < 5:
    recommendations.append(
        "Increase aeration"
    )

if ammonia > 1:
    recommendations.append(
        "Replace/filter water"
    )

if pH < 6.5:
    recommendations.append(
        "Add pH buffer"
    )

if turbidity > 5:
    recommendations.append(
        "Clean filtration system"
    )

if len(recommendations) == 0:
    recommendations.append(
        "System stable"
    )

# ---------------------------
# DISPLAY RESULTS
# ---------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.success(
        f"Predicted Health: {round(prediction,2)}"
    )

with col2:
    st.info(
        f"Sustainability Score: {round(sustainability_score,2)}"
    )

with col3:

    risk = "LOW"

    if ammonia > 1 or do < 5:
        risk = "HIGH"

    st.warning(
        f"Risk Level: {risk}"
    )

# ---------------------------
# RECOMMENDATIONS
# ---------------------------

st.subheader("AI Recommendations")
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2909/2909763.png",
    width=120
)

for rec in recommendations:
    st.write("•", rec)

# ---------------------------
# DIGITAL TWIN SIMULATION
# ---------------------------

st.subheader(
    "Digital Twin Simulation"
)

future = []

current_do = do

for i in range(20):

    current_do -= 0.05

    health = (
        0.3 * pH
        +
        0.4 * current_do
        -
        0.3 * ammonia
    )

    future.append(health)

st.subheader("📈 Real-Time Digital Twin")

chart_data = pd.DataFrame(
    future,
    columns=["System Health"]
)

st.line_chart(chart_data)