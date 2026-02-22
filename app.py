import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="AI Weather Alert System", layout="wide")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    files = [
        "bengaluru.csv","bombay.csv","delhi.csv",
        "hyderabad.csv","jaipur.csv","kanpur.csv",
        "nagpur.csv","pune.csv"
    ]
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["city"] = file.replace(".csv","")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()

df = load_data()
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time")

# ==============================
# LOAD MODELS
# ==============================
temp_model = joblib.load("temp_model.pkl")
rain_model = joblib.load("rain_model.pkl")
heat_model = joblib.load("heat_model.pkl")
disaster_model = joblib.load("disaster_model.pkl")

st.title("🚨 AI Extreme Weather Early Warning System")
st.write("Today: 22 Feb 2026")

city = st.selectbox("Select City", df["city"].unique())

city_df = df[df["city"] == city].copy()

# ==============================
# FEATURE ENGINEERING (SAME AS TRAINING!)
# ==============================
city_df["temp_lag_24"] = city_df["tempC"].shift(24)
city_df["humidity_lag_24"] = city_df["humidity"].shift(24)
city_df["rain_lag_24"] = city_df["precipMM"].shift(24)
city_df["wind_lag_24"] = city_df["windspeedKmph"].shift(24)

city_df["temp_roll_24"] = city_df["tempC"].rolling(24).mean()
city_df["humidity_roll_24"] = city_df["humidity"].rolling(24).mean()

latest = city_df.dropna().iloc[-1]

features = np.array([[
    latest["tempC"],
    latest["humidity"],
    latest["pressure"],
    latest["windspeedKmph"],
    latest["temp_lag_24"],
    latest["humidity_lag_24"],
    latest["rain_lag_24"],
    latest["wind_lag_24"],
    latest["temp_roll_24"],
    latest["humidity_roll_24"],
    latest["date_time"].month,
    latest["date_time"].hour,
    latest["date_time"].dayofweek
]])

# ==============================
# PREDICTIONS
# ==============================
pred_temp = temp_model.predict(features)[0]
rain_pred = rain_model.predict(features)[0]
heat_pred = heat_model.predict(features)[0]
disaster_pred = disaster_model.predict(features)[0]

# ==============================
# DISPLAY
# ==============================
st.metric("🌡 Tomorrow Temperature", f"{pred_temp:.2f} °C")

if heat_pred == 1:
    st.error("🔥 HEATWAVE ALERT")
elif rain_pred == 1:
    st.warning("🌧 Rain Expected Tomorrow")
else:
    st.success("✅ Normal Weather")

if disaster_pred == 1:
    st.error("🚨 EXTREME HEAT WARNING")
elif disaster_pred == 2:
    st.error("🚨 HEAVY RAIN WARNING")
elif disaster_pred == 3:
    st.error("🚨 STORM WARNING")

st.subheader("Temperature Trend")
st.line_chart(city_df.set_index("date_time")["tempC"])
