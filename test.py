import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
from tensorflow.python.framework import ops
from hazm import *
import json
import pickle

stemmer = LancasterStemmer()


with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            normalizer = Normalizer()
            normalizer.normalize(pattern)
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            docs_x.append(word)
            docs_y.append(intent["tags"])
        labels.append(intent["tags"])


    training = []
    output = []
    # print(docs_y, 'docs_y')
    output_empty = []

    # lemmatizer = Lemmatizer()
    print(words)
    # words = [lemmatizer.lemmatize(w) for w in words if w not in "؟"]
    print(docs_x)
    print(words)

    output_empty = [0 for _ in range(len(labels))]

    print(output_empty)

    for x, doc in enumerate(docs_x):
        bag = []
        print(x)
        print(doc)
        stemmer = Stemmer()
        wrds = [stemmer.stem(w) for w in doc]
        # print(wrds)
        for w in words:
            if w in doc:
                bag.append(1)
            else:
                bag.append(0)
        training.append(bag)
        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        output.append(output_row)
        # print(output_row)
    print(output, 'output')
    print(training, 'training')

    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)