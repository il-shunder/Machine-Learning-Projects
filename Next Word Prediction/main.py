import os
import random

import numpy as np
import pandas as pd
import requests
from keras import layers, losses, models, optimizers
from nltk.tokenize import RegexpTokenizer

NUMBER_OF_WORDS = 10
DATASET_FILE = "dataset.csv"
MODEL_PATH = "next_word_prediction.keras"


if not os.path.isfile(DATASET_FILE):
    url = "https://raw.githubusercontent.com/lutzhamel/fake-news/refs/heads/master/data/fake_or_real_news.csv"
    response = requests.get(url, allow_redirects=True)

    with open(DATASET_FILE, "wb") as file:
        file.write(response.content)

df = pd.read_csv(DATASET_FILE)

text = list(df["text"].values)
text = " ".join(text)

train_text = text[:300000].lower()

tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(train_text)

unique_tokens = np.unique(tokens)
words = {name: i for i, name in enumerate(unique_tokens)}

try:
    model = models.load_model(MODEL_PATH)
except ValueError:
    input_words = []
    next_words = []

    for i in range(len(tokens) - NUMBER_OF_WORDS):
        input_words.append(tokens[i : i + NUMBER_OF_WORDS])
        next_words.append(tokens[i + NUMBER_OF_WORDS])

    x_train = np.zeros((len(input_words), NUMBER_OF_WORDS, len(unique_tokens)))
    y_train = np.zeros((len(next_words), len(unique_tokens)))

    for i in range(len(input_words)):
        for j in range(NUMBER_OF_WORDS):
            x_train[i, j, words[input_words[i][j]]] = 1
        y_train[i, words[next_words[i]]] = 1

    model = models.Sequential(
        [
            layers.Input((NUMBER_OF_WORDS, len(unique_tokens))),
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(128),
            layers.Dense(len(unique_tokens)),
            layers.Activation(activation="linear"),
        ]
    )

    model.compile(
        loss=losses.CategoricalCrossentropy(from_logits=True),
        optimizer=optimizers.RMSprop(learning_rate=0.01),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, batch_size=128, epochs=30, shuffle=True)
    model.save(MODEL_PATH)
finally:
    if model:

        def predict_next_word(input_text, n_best):
            input_tokens = tokenizer.tokenize(input_text.lower())
            if len(input_tokens) == NUMBER_OF_WORDS:
                x_test = np.zeros((1, NUMBER_OF_WORDS, len(unique_tokens)))
                for i in range(NUMBER_OF_WORDS):
                    if input_tokens[i] in words:
                        index = words[input_tokens[i]]
                    else:
                        index = random.randint(0, len(words))
                    x_test[0, i, index] = 1
                predictions = model.predict(x_test)[0]
                predictions = np.argpartition(predictions, -n_best)[-n_best:]
                print([unique_tokens[i] for i in predictions])
            else:
                print(f"Input text must contain exactly {NUMBER_OF_WORDS} words")

        predict_next_word("He see the problem and decide to go there to", 5)
