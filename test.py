import nltk
import pickle
import numpy

with open("data.pickle", 'rb') as f:
    words, labels, training, output = pickle.load(f)

def bag_of_words(inputs, wrd):
    bags = [0 for _ in wrd]
    input_words = nltk.word_tokenize(inputs)
    for s in input_words:
        for i, w in enumerate(wrd):
            if w == s:
                bags[i] = 1
    return numpy.array(bags)


def chat_box():
    print("با من حرف بزن لعنتی!")
    while True:
        inputss = input("شما: ")
        if inputss == "quit":
            break

        print(bag_of_words(inputss, words))


chat_box()
