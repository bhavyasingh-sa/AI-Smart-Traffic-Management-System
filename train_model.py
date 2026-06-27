import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
df = pd.read_csv("dataset/cleaned_traffic_data.csv")
df["holiday"] = df["holiday"].fillna("NoHoliday")
df["date_time"] = pd.to_datetime(df["date_time"])
df["hour"] = df["date_time"].dt.hour
df["month"] = df["date_time"].dt.month
df["day"] = df["date_time"].dt.day
encoder = LabelEncoder()
df["weather_main"] = encoder.fit_transform(df["weather_main"])
X = df[
    [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "hour",
        "month",
        "day",
    ]
]
y = df["traffic_volume"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
print("Mean Absolute Error:", mae)
joblib.dump(model, "models/traffic_model.pkl")
print("\nModel saved successfully!")