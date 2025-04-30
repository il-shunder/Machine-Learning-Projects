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

    def bag_of_words(self, words, vocabulary):
        return [1 if word in words else 0 for word in vocabulary]

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
