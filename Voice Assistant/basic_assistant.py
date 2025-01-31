import json
import random

import nltk
import numpy as np
from keras import layers, losses, models
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")

MODEL_PATH = "chatbot_model.keras"


lemmatizer = WordNetLemmatizer()
data = json.loads(open("intents.json").read())
words = []
classes = []
documents = []
ignore_characters = ["?", "!", ".", ","]

for item in data["intents"]:
    for pattern in item["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        word_list = [word.lower() for word in word_list]
        words.extend(word_list)
        documents.append((word_list, item["tag"]))
        if item["tag"] not in classes:
            classes.append(item["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_characters]
words = sorted(set(words))

try:
    model = models.load_model(MODEL_PATH)
except ValueError:
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype="object")

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    model = models.Sequential(
        [
            layers.Input(shape=(len(train_x[0]),)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(len(train_y[0]), activation="linear"),
        ]
    )

    model.compile(optimizer="adam", loss=losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=8, verbose=1)
    model.save(MODEL_PATH)
finally:
    if model:

        def clean_up_sentence(sentence):
            sentence_words = nltk.word_tokenize(sentence)
            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
            return sentence_words

        def bag_of_words(sentence):
            sentence_words = clean_up_sentence(sentence)
            bag = [0] * len(words)
            for w in sentence_words:
                for i, word in enumerate(words):
                    if word == w:
                        bag[i] = 1
            return np.array(bag)

        def predict_class(sentence):
            bow = bag_of_words(sentence)
            prediction = model.predict(np.array([bow]))[0]
            index = np.argmax(prediction)
            return classes[index]

        def get_response(predicted_class, data_json):
            for item in data_json["intents"]:
                if item["tag"] == predicted_class:
                    result = random.choice(item["responses"])
                    break
            return result

        while True:
            message = input("Enter your message: ")
            prediction = predict_class(message)
            result = get_response(prediction, data)
            print(result)
