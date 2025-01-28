import json
import pickle
import random

import nltk
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

data = json.loads(open("requests.json").read())

words = []
classes = []
documents = []
ignore_characters = ["?", "!", ".", ","]

for request in data["requests"]:
    for pattern in request["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, request["tag"]))
        if request["tag"] not in classes:
            classes.append(request["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_characters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word.lower() in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype="object")

train_x = list(training[:, 0])
train_y = list(training[:, 1])

print(train_x)
print(train_y)
