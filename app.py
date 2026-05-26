from groq import Groq
from streamlit_autorefresh import st_autorefresh

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import random
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------
# GROQ API CONFIGURATION
# ---------------------------

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(
    api_key=GROQ_API_KEY
)

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
X_test_scaled = scaler.transform(X_test)

y_pred_test = model.predict(X_test_scaled)

model_accuracy = r2_score(
    y_test,
    y_pred_test
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
# PROJECT ANALYTICS
# ---------------------------

st.header("📊 Project Analytics Section")

# Analytics Metrics

analytics1, analytics2 = st.columns(2)

with analytics1:
    st.metric(
        "Model Accuracy (R² Score)",
        round(model_accuracy, 3)
    )

with analytics2:
    st.metric(
        "Dataset Size",
        len(data)
    )

# ---------------------------
# FEATURE IMPORTANCE
# ---------------------------

st.subheader("📌 Feature Importance Analysis")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)

# ---------------------------
# SUSTAINABILITY TREND
# ---------------------------

st.subheader("🌍 Sustainability Trend Analysis")

trend = []

base_score = sustainability_score

for i in range(30):

    base_score += random.uniform(-2, 2)

    trend.append(base_score)

trend_df = pd.DataFrame(
    trend,
    columns=["Sustainability Score"]
)

st.line_chart(trend_df)

# ---------------------------
# RISK DISTRIBUTION
# ---------------------------

st.subheader("⚠️ Risk Distribution")

risk_data = pd.DataFrame({
    "Risk Level": ["Low", "Medium", "High"],
    "Systems": [65, 25, 10]
})

st.bar_chart(
    risk_data.set_index("Risk Level")
)

# ---------------------------
# ANOMALY DETECTION
# ---------------------------

st.subheader("🚨 Anomaly Detection")

anomalies = []

if live_ammonia > 1.5:
    anomalies.append(
        "Critical ammonia spike detected"
    )

if live_do < 4:
    anomalies.append(
        "Low dissolved oxygen anomaly detected"
    )

if live_temp > 35:
    anomalies.append(
        "High temperature anomaly detected"
    )

if len(anomalies) == 0:

    st.success(
        "No anomalies detected"
    )

else:

    for anomaly in anomalies:
        st.error(anomaly)

# ---------------------------
# GEMINI AI CHATBOT
# ---------------------------

st.header("🤖 AURAVERDE AI Assistant")

user_question = st.text_input(
    "Ask the AI assistant about the system"
)

if user_question:

    prompt = f"""
    You are an AI assistant for the AURAVERDE Intelligent Aquaponics Platform.

    Current system conditions:

    pH: {round(live_pH,2)}
    Temperature: {round(live_temp,2)}
    Dissolved Oxygen: {round(live_do,2)}
    Turbidity: {round(live_turbidity,2)}
    Ammonia: {round(live_ammonia,2)}

    Sustainability Score:
    {round(sustainability_score,2)}

    Predicted Health:
    {round(prediction,2)}

    Risk Level:
    {risk}

    User Question:
    {user_question}

    Give a professional, concise, and technically accurate response.
    """

    try:

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )

        reply = chat_completion.choices[0].message.content

        st.write(reply)

    except Exception as e:

        st.error(f"Groq API Error: {e}")