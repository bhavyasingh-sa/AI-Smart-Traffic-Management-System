import pandas as pd
df = pd.read_csv("dataset/traffic_data.csv")
df["holiday"] = df["holiday"].fillna("NoHoliday")
df["date_time"] = pd.to_datetime(df["date_time"])
df["year"] = df["date_time"].dt.year
df["month"] = df["date_time"].dt.month
df["day"] = df["date_time"].dt.day
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.day_name()
print(df.head())
df.to_csv("dataset/cleaned_traffic_data.csv", index=False)
print("\nDataset processed successfully!")