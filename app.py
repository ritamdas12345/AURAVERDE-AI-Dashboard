from streamlit_autorefresh import st_autorefresh

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import random

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

# ---------------------------
# AUTO REFRESH
# ---------------------------

st_autorefresh(
    interval=3000,
    key="iot_simulation"
)

# ---------------------------
# LOAD DATA
# ---------------------------

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
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestRegressor(
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

# ---------------------------
# SHAP EXPLAINER
# ---------------------------

explainer = shap.TreeExplainer(model)

# ---------------------------
# SIDEBAR
# ---------------------------

st.sidebar.header("Input Parameters")

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2909/2909763.png",
    width=120
)

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
# REAL-TIME IoT SIMULATION
# ---------------------------

live_pH = pH + random.uniform(-0.3, 0.3)

live_temp = temperature + random.uniform(-1, 1)

live_do = do + random.uniform(-0.5, 0.5)

live_turbidity = turbidity + random.uniform(-0.5, 0.5)

live_ammonia = ammonia + random.uniform(-0.2, 0.2)

# ---------------------------
# DASHBOARD TITLE
# ---------------------------

st.markdown("""
# 🌱 AURAVERDE Intelligent Aquaponics Platform

### AI + IoT + Sustainability Analytics
""")

# ---------------------------
# LIVE IoT SENSOR FEED
# ---------------------------

st.subheader("📡 Real-Time IoT Sensor Feed")

sensor1, sensor2, sensor3 = st.columns(3)

with sensor1:
    st.metric(
        "Live pH",
        round(live_pH, 2)
    )

with sensor2:
    st.metric(
        "Live Temperature",
        round(live_temp, 2)
    )

with sensor3:
    st.metric(
        "Live Dissolved Oxygen",
        round(live_do, 2)
    )

sensor4, sensor5 = st.columns(2)

with sensor4:
    st.metric(
        "Live Turbidity",
        round(live_turbidity, 2)
    )

with sensor5:
    st.metric(
        "Live Ammonia",
        round(live_ammonia, 2)
    )

# ---------------------------
# CREATE INPUT DATAFRAME
# ---------------------------

input_data = pd.DataFrame({
    "pH": [live_pH],
    "temperature_C": [live_temp],
    "dissolved_oxygen_mg_L": [live_do],
    "turbidity_NTU": [live_turbidity],
    "ammonia_mg_L": [live_ammonia]
})

# ---------------------------
# PREDICTION
# ---------------------------

scaled_input = scaler.transform(input_data)

prediction = model.predict(
    scaled_input
)[0]

# ---------------------------
# SHAP VALUES
# ---------------------------

shap_values = explainer.shap_values(
    scaled_input
)

# ---------------------------
# SUSTAINABILITY SCORE
# ---------------------------

water_quality = (
    live_pH + live_do
) / 2

score = (
    0.7 * water_quality -
    0.3 * live_ammonia
)

sustainability_score = max(
    0,
    min(100, score * 10)
)

# ---------------------------
# AI RECOMMENDATIONS
# ---------------------------

recommendations = []

if live_do < 5:
    recommendations.append(
        "Increase aeration"
    )

if live_ammonia > 1:
    recommendations.append(
        "Replace/filter water"
    )

if live_pH < 6.5:
    recommendations.append(
        "Add pH buffer"
    )

if live_turbidity > 5:
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
        f"Predicted Health: {round(prediction, 2)}"
    )

with col2:
    st.info(
        f"Sustainability Score: {round(sustainability_score, 2)}"
    )

with col3:

    risk = "LOW"

    if live_ammonia > 1 or live_do < 5:
        risk = "HIGH"

    st.warning(
        f"Risk Level: {risk}"
    )

# ---------------------------
# AI RECOMMENDATIONS
# ---------------------------

st.subheader("🤖 AI Recommendations")

for rec in recommendations:
    st.write("•", rec)

# ---------------------------
# SHAP VISUALIZATION
# ---------------------------

st.subheader("🔍 Explainable AI Analysis")

fig_shap, ax = plt.subplots()

shap.summary_plot(
    shap_values,
    input_data,
    show=False
)

st.pyplot(fig_shap)

# ---------------------------
# DIGITAL TWIN SIMULATION
# ---------------------------

st.subheader("📈 Real-Time Digital Twin")

future = []

current_do = live_do

for i in range(20):

    current_do -= 0.05

    health = (
        0.3 * live_pH +
        0.4 * current_do -
        0.3 * live_ammonia
    )

    future.append(health)

chart_data = pd.DataFrame(
    future,
    columns=["System Health"]
)

st.line_chart(chart_data)
# ---------------------------
# AI CHATBOT ASSISTANT
# ---------------------------

st.subheader("🤖 AURAVERDE AI Assistant")
user_question = st.text_input(
    "Ask the AI assistant about the system"
)
if user_question:

    question = user_question.lower()

    if "oxygen" in question:
        st.write(
            "Dissolved oxygen is essential for fish survival and water quality stability."
        )

    elif "ammonia" in question:
        st.write(
            "High ammonia levels are harmful for aquatic life. Consider filtration or water replacement."
        )

    elif "ph" in question:
        st.write(
            "pH imbalance can affect nutrient absorption and aquatic ecosystem stability."
        )

    elif "temperature" in question:
        st.write(
            "Temperature influences fish metabolism and dissolved oxygen concentration."
        )

    elif "sustainability" in question:
        st.write(
            "The sustainability score is calculated using water quality and environmental stability indicators."
        )

    elif "health" in question:
        st.write(
            f"The current predicted system health is {round(prediction,2)}."
        )

    elif "risk" in question:
        st.write(
            f"The current system risk level is {risk}."
        )

    else:
        st.write(
            "System operating normally. Please ask about pH, oxygen, ammonia, sustainability, or risk."
        )