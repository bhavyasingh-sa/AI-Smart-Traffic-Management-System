
import streamlit as st
import pandas as pd
import joblib
import google.genai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
st.set_page_config(
    page_title="AI Smart Traffic Management",
    layout="wide"
)
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/cleaned_traffic_data.csv")
    df["holiday"] = df["holiday"].fillna("NoHoliday")
    return df
@st.cache_resource
def load_model():
    import os
    from sklearn.ensemble import RandomForestRegressor
    model_path = "models/traffic_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    df = pd.read_csv("dataset/cleaned_traffic_data.csv")
    df["holiday"] = df["holiday"].fillna("NoHoliday")
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
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    return model
@st.cache_resource
def create_encoder(df):
    encoder = LabelEncoder()
    encoder.fit(df["weather_main"])
    return encoder
@st.cache_resource
def build_search_index(df):
    df["search_text"] = (
    df["weather_main"].astype(str)
    + " "
    + df["weather_description"].astype(str)
    + " "
    + df["day_of_week"].astype(str)
    + " "
    + df["holiday"].astype(str)
    + " "
    + df["hour"].astype(str)
)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["search_text"])
    return vectorizer, tfidf_matrix
def predict_traffic(
    model,
    encoder,
    temp,
    rain,
    snow,
    clouds,
    weather,
    hour,
    month,
    day
):
    weather_encoded = encoder.transform([weather])[0]
    input_df = pd.DataFrame([[
        temp,
        rain,
        snow,
        clouds,
        weather_encoded,
        hour,
        month,
        day
    ]],
    columns=[
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "hour",
        "month",
        "day"
    ])
    prediction = model.predict(input_df)[0]
    return prediction
def find_similar_cases(
    query,
    df,
    vectorizer,
    tfidf_matrix
):
    query_vector = vectorizer.transform([query])
    similarity = cosine_similarity(
        query_vector,
        tfidf_matrix
    )
    top = similarity[0].argsort()[-5:][::-1]
    return df.iloc[top]

def get_ai_explanation(
    prediction,
    congestion,
    signal,
    similar_records
):
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

    prompt = f"""
You are an AI Traffic Management Assistant.

Current Prediction

Traffic Volume: {prediction:.0f}

Congestion: {congestion}

Green Signal Recommendation: {signal}

Historical Traffic Records

{similar_records.to_string(index=False)}

Explain:

1. Why congestion is occurring.
2. How the historical records support this prediction.
3. Why the recommended signal timing is appropriate.

Keep the answer under 150 words.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
def show_dashboard():
    st.title("AI-Based Smart Traffic Management System")
    st.caption(
    "Traffic Congestion Prediction using Information Retrieval and Machine Learning"
)
    st.sidebar.header("Traffic Conditions")
    temp = st.sidebar.number_input("Temperature", value=289.0)
    rain = st.sidebar.number_input("Rain", value=0.0)
    snow = st.sidebar.number_input("Snow", value=0.0)
    clouds = st.sidebar.slider("Cloud Cover", 0, 100, 50)
    weather = st.sidebar.selectbox(
        "Weather",
        sorted(df["weather_main"].unique())
    )
    hour = st.sidebar.slider("Hour", 0, 23, 9)
    month = st.sidebar.slider("Month", 1, 12, 10)
    day = st.sidebar.slider("Day", 1, 31, 2)
    st.divider()

    if st.button(
        "🚦 Predict Traffic",
        use_container_width=True
    ):
        with st.container(border=True):
            st.subheader("Prediction Summary")
            prediction = predict_traffic(
                model,
                encoder,
                temp,
                rain,
                snow,
                clouds,
                weather,
                hour,
                month,
                day
            )
            if prediction < 2000:
                congestion = "Low"
                signal = "30 sec"
            elif prediction < 4000:
                congestion = "Medium"
                signal = "60 sec"
            else:
                congestion = "High"
                signal = "90 sec"
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Traffic Volume", int(prediction))
            with col2:
                st.metric("Congestion", congestion)
            with col3:
                st.metric("Green Signal", signal)
            progress = min(int(prediction / 6000 * 100), 100)
            st.progress(progress)
            st.caption(f"Current Traffic Load: {progress}%")
        with st.container(border=True):
            st.subheader("Historical Traffic Records")
            query = f"{weather} {hour}"
            similar = find_similar_cases(
                query,
                df,
                vectorizer,
                tfidf_matrix
            )
            st.dataframe(
                similar[
                    [
                        "date_time",
                        "weather_main",
                        "weather_description",
                        "hour",
                        "traffic_volume"
                    ]
                ],
                use_container_width=True
            )

            st.divider()

            st.subheader("🤖 AI Explanation")

            with st.spinner("Generating explanation..."):

                explanation = get_ai_explanation(
                    prediction,
                    congestion,
                    signal,
                    similar
                )

            st.write(explanation)

df = load_data()
model = load_model()
encoder = create_encoder(df)
vectorizer, tfidf_matrix = build_search_index(df)
show_dashboard()
with st.container(border=True):
    st.subheader("Traffic Analytics")
    st.write("### Average Traffic by Hour")
    hourly = df.groupby("hour")["traffic_volume"].mean()
    st.line_chart(hourly)
    st.write("### Weather Distribution")
    weather_counts = df["weather_main"].value_counts()
    st.bar_chart(weather_counts)
    st.write("### Average Traffic by Month")
    monthly = df.groupby("month")["traffic_volume"].mean()
    st.line_chart(monthly)