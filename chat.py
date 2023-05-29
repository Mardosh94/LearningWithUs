import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer

class ChatBot:
    def __init__(self, intents_file):
        with open(intents_file) as file:
            self.intents = json.load(file)

        self.stemmer = LancasterStemmer()
        nltk.download('punkt')

        self.words = []
        self.labels = []
        self.docs_x = []
        self.docs_y = []

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                wrds = nltk.word_tokenize(pattern)
                self.words.extend(wrds)
                self.docs_x.append(wrds)
                self.docs_y.append(intent['tag'])

            if intent['tag'] not in self.labels:
                self.labels.append(intent['tag'])

        self.words = [self.stemmer.stem(w.lower()) for w in self.words if w != "?"]
        self.words = sorted(list(set(self.words)))
        self.labels = sorted(self.labels)

        self.training = []
        self.output = []
        self.out_empty = [0 for _ in range(len(self.labels))]

        for x, doc in enumerate(self.docs_x):
            bag = []
            wrds = [self.stemmer.stem(w.lower()) for w in doc]

            for w in self.words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = self.out_empty[:]
            output_row[self.labels.index(self.docs_y[x])] = 1

            self.training.append(bag)
            self.output.append(output_row)

        self.training = np.array(self.training)
        self.output = np.array(self.output)

        tf.compat.v1.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(self.training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.output[0]), activation='softmax')
        net = tflearn.regression(net)

        self.model = tflearn.DNN(net)
        self.model.fit(self.training, self.output, n_epoch=1000, batch_size=8, show_metric=True)
        self.model.save('model.tflearn')

    def bag_of_words(self, s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return np.array(bag)

    def get_response(self, user_input):
        results = self.model.predict([self.bag_of_words(user_input, self.words)])[0]
        results_index = np.argmax(results)
        tag = self.labels[results_index]

        for tg in self.intents["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return random.choice(responses)

    def chat(self):
        print("Start talking with the bot (type quit to stop)!")
        while True:
            inp = input("You: ")
            if inp.lower() == "quit":
                break

            results = self.model.predict([self.bag_of_words(inp, self.words)])[0]
            results_index = np.argmax(results)
            tag = self.labels[results_index]

            for tg in self.intents["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))


