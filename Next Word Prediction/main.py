import os

import numpy as np
import pandas as pd
import requests
from nltk.tokenize import RegexpTokenizer

NUMBER_OF_WORDS = 10
DATASET_FILE = "dataset.csv"

if not os.path.isfile(DATASET_FILE):
    url = "https://raw.githubusercontent.com/lutzhamel/fake-news/refs/heads/master/data/fake_or_real_news.csv"
    response = requests.get(url, allow_redirects=True)

    with open(DATASET_FILE, "wb") as file:
        file.write(response.content)

df = pd.read_csv(DATASET_FILE)

text = list(df["text"].values)
text = " ".join(text)

train_text = text[:100].lower()

tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(train_text)

unique_tokens = np.unique(tokens)

input_words = []
next_words = []

for i in range(len(tokens) - NUMBER_OF_WORDS):
    input_words.append(tokens[i : i + NUMBER_OF_WORDS])
    next_words.append(tokens[i + NUMBER_OF_WORDS])

print(tokens)
print()
print(input_words)
print()
print(next_words)
print()
