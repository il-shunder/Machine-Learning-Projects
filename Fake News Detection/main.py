import os

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

DATASET_FILE = "dataset.csv"
TEST_TEXT_FILE = "test_text.txt"


if not os.path.isfile(DATASET_FILE):
    url = "https://raw.githubusercontent.com/lutzhamel/fake-news/refs/heads/master/data/fake_or_real_news.csv"
    response = requests.get(url, allow_redirects=True)

    with open(DATASET_FILE, "wb") as file:
        file.write(response.content)

df = pd.read_csv(DATASET_FILE)

df["is_fake"] = df["label"].apply(lambda x: 1 if x == "FAKE" else 0)

X, y = df["text"], df["is_fake"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text data to TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Add test data
with open(TEST_TEXT_FILE, "w", encoding="utf-8") as file:
    file.write(X_test.iloc[15])


def detect_fake_news():
    if os.path.isfile(TEST_TEXT_FILE):
        with open(TEST_TEXT_FILE, "r", encoding="utf-8") as file:
            test_text = file.read()

        test_text_tfidf = tfidf_vectorizer.transform([test_text])
        prediction = clf.predict(test_text_tfidf)[0]
        if prediction:
            print("The news is fake")
        else:
            print("The news is real")


detect_fake_news()
