import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
model = joblib.load("models/traffic_model.pkl")
df = pd.read_csv("dataset/cleaned_traffic_data.csv")
df["holiday"] = df["holiday"].fillna("NoHoliday")
encoder = LabelEncoder()
encoder.fit(df["weather_main"])
print("=== Traffic Congestion Prediction ===")
temp = float(input("Temperature (Kelvin): "))
rain = float(input("Rain in last hour: "))
snow = float(input("Snow in last hour: "))
clouds = int(input("Cloud Percentage (0-100): "))
weather = input("Weather (Clouds, Rain, Clear, Mist, Snow): ")
hour = int(input("Hour (0-23): "))
month = int(input("Month (1-12): "))
day = int(input("Day of Month: "))
weather_encoded = encoder.transform([weather])[0]
input_data = pd.DataFrame([[
    temp,
    rain,
    snow,
    clouds,
    weather_encoded,
    hour,
    month,
    day
]], columns=[
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "hour",
    "month",
    "day"
])
prediction = model.predict(input_data)[0]
print(f"\nPredicted Traffic Volume: {prediction:.0f}")
if prediction < 2000:
    congestion = "LOW"
    signal = 30
elif prediction < 4000:
    congestion = "MEDIUM"
    signal = 60
else:
    congestion = "HIGH"
    signal = 90
print(f"Congestion Level: {congestion}")
print(f"Recommended Green Signal Time: {signal} seconds")