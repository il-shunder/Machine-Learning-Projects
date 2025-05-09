import json
import os
import random
from turtle import forward

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader, TensorDataset

# nltk.download("punkt_tab", quiet=True)
# nltk.download("punkt", quiet=True)
# nltk.download("wordnet", quiet=True)


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)


class ChatbotAssistant(nn.Module):
    def __init__(self, intents_path: str, method_mappings: dict = {}):
        self.model = None
        self.intents_path = intents_path
        self.method_mappings = method_mappings

        self.lemmatizer = WordNetLemmatizer()

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.X = None
        self.y = None

    def tokenize_and_lemmatize(self, text):
        words = nltk.word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        if os.path.exists(self.intents_path):
            intents_data = json.loads(open(self.intents_path).read())

            for intent in intents_data["intents"]:
                if intent["tag"] not in self.intents:
                    self.intents.append(intent["tag"])
                    self.intents_responses[intent["tag"]] = intent["responses"]

                for pattern in intent["patterns"]:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent["tag"]))

            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=lr)

        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss

            print(f"Epoch: {epoch + 1}, Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, "w") as f:
            json.dump({"input_size": self.X.shape[1], "output_size": len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, "r") as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions["input_size"], dimensions["output_size"])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, message):
        words = self.tokenize_and_lemmatize(message)
        bag = self.bag_of_words(message)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

            predicted_class_index = torch.argmax(predictions, dim=1).item()
            prediction_intent = self.intents[predicted_class_index]

            if self.method_mappings:
                if prediction_intent in self.method_mappings:
                    self.method_mappings[prediction_intent]()

            if self.intents_responses[prediction_intent]:
                return random.choice(self.intents_responses[prediction_intent])
            else:
                return None
