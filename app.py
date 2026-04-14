import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# load model & columns
model = joblib.load("traffic_model.pkl")
training_columns = joblib.load("feature_order.pkl")

# helper functions
def process_date(date):
    dow = date.weekday()
    is_weekend = 1 if dow in [5,6] else 0
    return dow, is_weekend

def encode_weather(data, weather):
    weather_cols = [
        'weather_main_Clouds','weather_main_Drizzle','weather_main_Fog',
        'weather_main_Haze','weather_main_Mist','weather_main_Rain',
        'weather_main_Smoke','weather_main_Snow','weather_main_Squall',
        'weather_main_Thunderstorm'
    ]

    weather_dict = {col: 0 for col in weather_cols}

    col_name = f"weather_main_{weather}"

    if col_name in weather_dict:
        weather_dict[col_name] = 1

    for col, val in weather_dict.items():
        if col in data.columns:
            data.at[0, col] = val

def classify(pred):
    if pred < 2000:
        return "Low"
    elif pred < 4000:
        return "Medium"
    else:
        return "High"

def traffic_light(pred):
    if pred < 2000:
        return {
            "green": 20,
            "red": 60,
            "note": "Low traffic – balanced timing"
        }
    elif pred < 4000:
        return {
            "green": 30,
            "red": 50,
            "note": "Moderate traffic – slight priority"
        }
    else:
        return {
            "green": 35,
            "red": 45,
            "note": "High traffic – prioritized flow"
        }

def generate_message(temp, weather, status, hour):
    is_day = 6 <= hour <= 18

    if weather == "Thunderstorm":
        return "⛈️ Thunderstorm alert! Drive carefully."

    if weather == "Snow":
        return "❄️ Snowy roads ahead, drive slow!"

    if weather == "Rain":
        return "🌧️ Rainy conditions, expect slower traffic."

    if weather in ["Fog", "Mist", "Haze"]:
        return "🌫️ Low visibility, drive carefully."

    if temp > 35:
        if is_day:
            return "🔥 Extremely hot day, stay hydrated and patient on the road."
        else:
            return "🌙 Warm night, roads may feel calmer."

    if temp > 30:
        if is_day:
            return "☀️ Quite hot outside, traffic might feel more exhausting."
        else:
            return "🌙 Mild warm night, comfortable for driving."

    if temp < 10:
        return "🥶 Cold weather, drive carefully and stay warm."

    if status == "High":
        if is_day:
            return "🚗 Heavy daytime traffic, expect delays."
        else:
            return "🌙 Busy night traffic, possibly due to events or travel."

    if status == "Medium":
        return "🚙 Moderate traffic, stay alert."

    # ✨ DEFAULT
    if is_day:
        return "✨ Smooth traffic, enjoy your daytime drive!"
    else:
        return "🌙 Calm night traffic, have a safe trip!"

# page setting
st.set_page_config(page_title="Mettraff", layout="wide")

# css styling
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-color: #e8ecef;
}

h1 {
    color: #2c3e50;
    text-align: center;
    font-weight: 600;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #8aa7bc;
    color: white;
    margin-bottom: 15px;
    font-weight: 500;
}

.block-container {
    padding-top: 2rem;
}

div[data-testid="column"] {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.stButton>button {
    background-color: #6c8ea4;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #557a91;
    color: white;
}

# labels
label {
    color: #2c3e50 !important;
    font-weight: 500;
}

.stSuccess {
    background-color: #d4edda !important;
    color: #155724 !important;
    border-radius: 10px;
}

.stInfo {
    background-color: #d1ecf1 !important;
    color: #0c5460 !important;
    border-radius: 10px;
}

.stWarning {
    background-color: #fff3cd !important;
    color: #856404 !important;
    border-radius: 10px;
}

@media (prefers-color-scheme: dark) {

    .stApp {
        background-color: #1e1e1e;
    }

    div[data-testid="column"] {
        background-color: #2b2b2b;
    }

    h1, label {
        color: #ffffff !important;
    }

    .card {
        background-color: #3a556a;
        color: white;
    }

    .stButton>button {
        background-color: #4a6b82;
        color: white;
    }

    .stButton>button:hover {
        background-color: #3a556a;
    }

    .stSuccess {
        background-color: #1e4620 !important;
        color: #b7f5c5 !important;
    }

    .stInfo {
        background-color: #1e3a46 !important;
        color: #bdefff !important;
    }

    .stWarning {
        background-color: #4a3b1e !important;
        color: #ffe7a3 !important;
    }
            
    .subtitle {
    font-size: 15px;
    text-align: center;
    opacity: 0.8;
    line-height: 1.6;
    letter-spacing: 0.5px;
    }
}

/* hide anchor link icon */
a[href^="#"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)


st.title("MetTraff")
st.markdown("""
<div class="card">
    <div class="subtitle">
        <em>Beat the traffic before it beats you! Getting stuck is so last season.</em>
    </div>
</div>
""", unsafe_allow_html=True)

# layout
left, right = st.columns([1,1])

# input
with left:
    st.markdown("### Input")

    temp = st.number_input("Temperature (°C)", value=25)

    weather = st.selectbox("Weather", [
        "Clouds","Drizzle","Fog","Haze","Mist",
        "Rain","Smoke","Snow","Squall","Thunderstorm"
    ])

    date = st.date_input("Date")
    time = st.time_input("Time")

    predict_btn = st.button("Predict Traffic")

# output
with right:
    st.markdown("### Result")

    if predict_btn:

        hour = time.hour
        dow, is_weekend = process_date(date)

        temp_kelvin = temp + 273.15
        rush = 1 if hour in [7,8,9,16,17,18] else 0

        data = pd.DataFrame(0.0, index=[0], columns=training_columns)

        data.at[0, 'temp'] = temp_kelvin
        data.at[0, 'hour'] = hour
        data.at[0, 'day_of_week'] = dow
        data.at[0, 'month'] = date.month

        data.at[0, 'is_weekend'] = is_weekend
        data.at[0, 'is_rush_hour'] = rush

        encode_weather(data, weather)

        pred = model.predict(data)[0]

        if pred < 0:
           pred = 10 + abs(pred) % 25

        status = classify(pred)
        lamp = traffic_light(pred)
        message = generate_message(temp, weather, status, hour)

        st.success(f"Traffic Volume: {int(pred)} kendaraan")
        st.info(f"Status: {status}")

        c1, c2 = st.columns(2)
        c1.metric("🟢 Green", f"{lamp['green']}s")
        c2.metric("🔴 Red", f"{lamp['red']}s")

        st.warning(message)

# sidebar 
with st.sidebar:
    st.header("About")
    st.write("MetTraff: Targeted Feature Expansion of Meteorological Data for Dynamic Traffic Flow Prediction and Optimization.")
