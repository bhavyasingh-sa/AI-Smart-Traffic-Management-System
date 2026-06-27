import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("dataset/cleaned_traffic_data.csv")
df["holiday"] = df["holiday"].fillna("NoHoliday")
df["search_text"] = (
    df["weather_main"].astype(str)
    + " "
    + df["weather_description"].astype(str)
    + " "
    + df["day_of_week"].astype(str)
    + " "
    + df["holiday"].astype(str)
)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["search_text"])
query = input("Enter your search query: ")
query_vector = vectorizer.transform([query])
similarity = cosine_similarity(query_vector, tfidf_matrix)
top_indices = similarity[0].argsort()[-5:][::-1]
print("\nTop 5 Matching Records\n")
for index in top_indices:
    print("--------------------------------------")
    print(df.iloc[index][
        [
            "weather_main",
            "weather_description",
            "day_of_week",
            "holiday",
            "traffic_volume",
        ]
    ])
    print("Similarity Score:", similarity[0][index])