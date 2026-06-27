import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv("dataset/cleaned_traffic_data.csv")
df["search_text"] = (
    df["weather_main"].astype(str) + " " +
    df["weather_description"].astype(str) + " " +
    df["day_of_week"].astype(str) + " " +
    df["holiday"].astype(str)
)
print(df["search_text"].head())
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["search_text"])
print("\nTF-IDF Matrix Shape:")
print(tfidf_matrix.shape)
print("\nVocabulary Size:")
print(len(vectorizer.vocabulary_))