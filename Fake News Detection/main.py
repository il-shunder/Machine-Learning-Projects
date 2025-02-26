import os

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

DATASET_FILE = "dataset.csv"


if not os.path.isfile(DATASET_FILE):
    url = "https://raw.githubusercontent.com/lutzhamel/fake-news/refs/heads/master/data/fake_or_real_news.csv"
    response = requests.get(url, allow_redirects=True)

    with open(DATASET_FILE, "wb") as file:
        file.write(response.content)

df = pd.read_csv(DATASET_FILE)

df["is_fake"] = df["label"].apply(lambda x: 1 if x == "FAKE" else 0)

X, y = df["text"], df["is_fake"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
