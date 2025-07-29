# train_models.py

import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from reddit_scraper import RedditScraper

def load_and_filter(path, label_col_correct):
    df = pd.read_excel(path)
    df = df[df[label_col_correct] == 'Yes']
    return df.dropna(subset=["body"])

def prepare_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])

def train_model(df, label_type, model_filename):
    X = df["body"]
    y = df[label_type]
    pipeline = prepare_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f"✅ Trained {model_filename} with accuracy: {accuracy:.4f}")

    joblib.dump(pipeline, f"models/{model_filename}")

def main():
    os.makedirs("models", exist_ok=True)

    tasks = [
        {
            "path": "data/training_data/sentiment_training_data.xlsx",
            "label_col_correct": "sentiment_label_correct",
            "label_type": "sentiment_label",
            "model_filename": "sentiment_model.pkl"
        },
        {
            "path": "data/training_data/opinion_training_data.xlsx",
            "label_col_correct": "opinion_label_correct",
            "label_type": "opinion_label",
            "model_filename": "opinion_model.pkl"
        },
        {
            "path": "data/training_data/plausibility_training_data.xlsx",
            "label_col_correct": "plausibility_label_correct",
            "label_type": "plausibility_label",
            "model_filename": "plausibility_model.pkl"
        }
    ]

    for task in tasks:
        df = load_and_filter(task["path"], task["label_col_correct"])
        train_model(df, task["label_type"], task["model_filename"])

    print("🎉 All models trained and saved.")

if __name__ == "__main__":
    main()
