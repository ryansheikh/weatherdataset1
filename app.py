import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

st.set_page_config(page_title="AI Weather Alert System", layout="wide")

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

# Load models
temp_model = joblib.load("temp_model.pkl")
rain_model = joblib.load("rain_model.pkl")
heat_model = joblib.load("heat_model.pkl")
disaster_model = joblib.load("disaster_model.pkl")

st.title("🚨 AI Extreme Weather Early Warning System")
st.write("Today's Date: 22 Feb 2026")

city = st.selectbox("Select City", df["city"].unique())
latest = df[df["city"] == city].iloc[-1]

features = [
    latest["tempC"], latest["humidity"], latest["pressure"],
    latest["windspeedKmph"],
    latest["tempC"], latest["humidity"],
    latest["precipMM"], latest["windspeedKmph"],
    latest["tempC"], latest["humidity"],
    datetime.now().month,
    datetime.now().hour,
    datetime.now().weekday()
]

features = np.array(features).reshape(1, -1)

pred_temp = temp_model.predict(features)[0]
rain_pred = rain_model.predict(features)[0]
heat_pred = heat_model.predict(features)[0]
disaster_pred = disaster_model.predict(features)[0]

st.metric("🌡 Predicted Tomorrow Temperature (°C)", round(pred_temp,2))

if heat_pred == 1:
    st.error("🔥 HEATWAVE ALERT!")
elif rain_pred == 1:
    st.warning("🌧 Rain Expected Tomorrow")
else:
    st.success("✅ Normal Weather Expected")

if disaster_pred == 1:
    st.error("🚨 EXTREME HEAT DISASTER WARNING")
elif disaster_pred == 2:
    st.error("🚨 HEAVY RAIN WARNING")
elif disaster_pred == 3:
    st.error("🚨 STORM WARNING")

st.line_chart(df[df["city"]==city].set_index("date_time")["tempC"])