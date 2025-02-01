import json
import os
import random

import nltk
import numpy as np
from keras import layers, losses, models
from nltk.stem import WordNetLemmatizer


class BasicAssistant:
    MODEL_PATH = "basic_assistant.keras"
    IGNORE_CHARACTERS = ["?", "!", ".", ","]

    def __init__(self, intents_path: str, method_mappings: dict = {}):
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)

        self.words = []
        self.intents = []
        self.documents = []
        self.training_data = []

        if os.path.exists(intents_path):
            self.intents_data = json.loads(open(intents_path).read())
            self._process_intents_data()
        else:
            raise FileNotFoundError

        self.lemmatizer = WordNetLemmatizer()
        self.model = self._load_model()
        self.method_mappings = method_mappings

    def _process_intents_data(self):
        for item in self.intents_data["intents"]:
            if item["tag"] not in self.intents:
                self.intents.append(item["tag"])

            for pattern in item["patterns"]:
                pattern_words = nltk.word_tokenize(pattern)
                pattern_words = [word.lower() for word in pattern_words]
                self.words.extend(pattern_words)
                self.documents.append((pattern_words, item["tag"]))

        self.words = [self.lemmatizer.lemmatize(word) for word in self.words if word not in self.IGNORE_CHARACTERS]
        self.words = sorted(set(self.words))

    def _load_model(self):
        try:
            model = models.load_model(self.MODEL_PATH)
        except ValueError:
            train_x, train_y = self._get_training_data()

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
            model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
            model.save(self.MODEL_PATH)
        finally:
            return model

    def _get_training_data(self):
        empty_output = [0] * len(self.intents)

        for document in self.documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            row_output = list(empty_output)
            row_output[self.intents.index(document[1])] = 1
            self.training_data.append([bag, row_output])

        random.shuffle(self.training_data)
        self.training_data = np.array(self.training_data, dtype="object")

        train_x = list(self.training_data[:, 0])
        train_y = list(self.training_data[:, 1])

        return train_x, train_y

    def _predict_intent(self, sentence: str):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        bag_of_words = [0] * len(self.words)

        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag_of_words[i] = 1

        bag_of_words = np.array([bag_of_words])

        prediction = self.model.predict(bag_of_words, verbose=0)[0]
        index = np.argmax(prediction)

        return self.intents[index]

    def get_response(self, sentence: str):
        predicted_intent = self._predict_intent(sentence)

        try:
            if predicted_intent in self.method_mappings:
                return self.method_mappings[predicted_intent]()

            for intent in self.intents_data["intents"]:
                if intent["tag"] == predicted_intent:
                    return random.choice(intent["responses"])
        except IndexError:
            return "Sorry, I don't understand you. Please try again."
