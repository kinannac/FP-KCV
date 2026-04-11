import gradio as gr
import joblib
import pandas as pd
from datetime import datetime

# === load ===
model = joblib.load("model.pkl")
training_columns = joblib.load("columns.pkl")
holiday_dict = joblib.load("holiday_dict.pkl")

# === helper ===
def process_date(day, month, year):
    date = datetime(year, month, day)
    dow = date.weekday()
    is_weekend = 1 if dow in [5,6] else 0
    return dow, is_weekend

def encode_weather(data, weather):
    col = f"weather_main_{weather}"
    if col in data.columns:
        data.at[0, col] = 1

def is_holiday_manual(day, month):
    holidays = [(1,1), (7,4), (12,25)]
    return 1 if (month, day) in holidays else 0

def classify(pred):
    if pred < 2000:
        return "Low"
    elif pred < 4000:
        return "Medium"
    else:
        return "High"

def traffic_light(pred):
    if pred < 2000:
        return {"green": 30, "red": 60}
    elif pred < 4000:
        return {"green": 45, "red": 45}
    else:
        return {"green": 60, "red": 30}

def generate_message(temp, weather, status, is_holiday, hour):
    weather = weather.lower()
    is_day = 6 <= hour <= 18

    if weather == "thunderstorm":
        return "⛈️ Thunderstorm alert! Drive carefully."

    if weather == "snow":
        return "❄️ Snowy roads ahead, drive slow!"

    if weather == "rain":
        return "🌧️ Rainy conditions, expect slower traffic."

    if weather in ["fog","mist","haze"]:
        return "🌫️ Low visibility, drive carefully."

    if temp > 35:
        return "🔥 Super hot day, stay hydrated!"

    if temp < 10:
        return "🥶 Cold weather, stay safe!"

    if is_holiday:
        return "🎉 Holiday detected, traffic pattern may change!"

    if status == "High":
        return "🚗 Heavy traffic, be patient!"

    return "✨ Smooth ride!"

# === MAIN ===
def predict(temp, weather, day, month, year, hour):

    dow, is_weekend = process_date(day, month, year)

    date_str = f"{year}-{month:02d}-{day:02d}"
    is_holiday = holiday_dict.get(date_str, is_holiday_manual(day, month))

    rush = 1 if hour in [7,8,9,16,17,18] else 0

    # convert C → K
    temp_kelvin = temp + 273.15

    data = pd.DataFrame(0.0, index=[0], columns=training_columns)

    # isi data
    data.at[0, 'temp'] = temp_kelvin
    data.at[0, 'hour'] = hour
    data.at[0, 'day'] = dow
    data.at[0, 'month'] = month

    data.at[0, 'is_weekend'] = is_weekend
    data.at[0, 'is_rush_hour'] = rush
    data.at[0, 'is_holiday'] = is_holiday

    # weather
    encode_weather(data, weather)

    data = data.fillna(0)

    pred = model.predict(data)[0]

    status = classify(pred)
    lamp = traffic_light(pred)
    message = generate_message(temp, weather, status, is_holiday, hour)

    color = {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🔴"
    }

    return f"""
    ### 🚗 Traffic Result

    **Volume:** {int(pred)}  
    **Status:** {color[status]} {status}  

    🚦 Green: {lamp['green']} sec  
    🛑 Red: {lamp['red']} sec  

    💬 {message}
    """

# === UI ===
css = """
body {
    background: linear-gradient(to right, #ffe4ec, #ffc1cc);
    font-family: 'Arial', sans-serif;
}

.gradio-container {
    border-radius: 15px;
}

button {
    background-color: #ff4d88 !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: bold;
}

textarea {
    border-radius: 10px !important;
}
"""

with gr.Blocks() as demo:

    # TITLE
    gr.Markdown("# MetTraff: Traffic Prediction AI")
    gr.Markdown("Smart traffic prediction with weather & date awareness ✨")

    with gr.Row():
        with gr.Column():
            temp = gr.Number(label="🌡 Temperature (°C)")
            weather = gr.Dropdown([
                "Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
                "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"
            ], label="🌤 Weather")

        with gr.Column():
            day = gr.Number(label="📅 Day")
            month = gr.Number(label="📆 Month")
            year = gr.Number(label="📆 Year")
            hour = gr.Number(0, 23,label="⏰ Hour")

    btn = gr.Button("🚀 Predict Traffic")

    output = gr.Markdown("### 📊 Result will appear here...")

    # FUNCTION CONNECT
    btn.click(
        fn=predict,
        inputs=[temp, weather, day, month, year, hour],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        css=css, 
        theme=gr.themes.Soft(primary_hue="pink")
    )