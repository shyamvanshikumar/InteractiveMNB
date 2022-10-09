from operator import itemgetter
from typing import List
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

classLabels = ["Company", "Education Institution", "Artist", "Ahtlete", "Office Holder", 
                "Mean of Transportation", "Building", "Natural Place"]

class MNBmodel:
    prevDoc = ""
    def __init__(self):
        corpus = pd.read_csv("./data/dbpedia_8K.csv")
        corpus.drop(columns=["title"], inplace=True)
        print(corpus.head())
        self.corpus = corpus

        initialVec = CountVectorizer(stop_words='english', max_features=100)
        initialData = initialVec.fit(corpus["content"])
        initialVoc = initialData.vocabulary_
        #initialVoc

        vocabulary = list(initialVoc.keys())#Make any changes in the vocabulary
        self.vocabulary = dict(zip(vocabulary, list(range(len(vocabulary)))))
        self.word_weight = np.ones(len(vocabulary))
    
    def getTrainTestData(self):
        changeVec = CountVectorizer(stop_words='english', vocabulary=self.vocabulary)
        vecData = changeVec.fit_transform(self.corpus["content"]).toarray()
        label = self.corpus["label"].to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(vecData, label, test_size=0.2, shuffle=True)

    def train(self):
        Fnc = np.zeros((8,len(self.vocabulary)))
        Prc = np.zeros((8,1))
        for i in range(8):
            indices = (self.y_train == i).astype(int)
            Fnc[i] = indices @ self.X_train
            Prc[i] = np.sum(indices)/len(self.y_train)
        
        print(Fnc.shape, self.word_weight.shape)
        Prwc = (Fnc + self.word_weight) / (np.sum(Fnc, axis=1) + np.sum(self.word_weight)).reshape((8,1))
        self.logPrwc = np.log(Prwc)
        self.logPrc = np.log(Prc)
    
    def test(self):
        y_pred = np.zeros(self.y_test.shape)
        for i in range(len(self.y_test)):
            doc = self.X_test[i]
            logPrcdi = np.sum((doc * self.logPrwc), axis=1).reshape((8,1)) + self.logPrc
            y_pred[i] = np.argmax(logPrcdi)

        self.acc = round(accuracy_score(self.y_test, y_pred), 2)
        print(self.acc)
    
    def predict(self, txtDoc=None):
        if txtDoc==None:
            txtDoc = self.prevDoc
        else:
            self.prevDoc = txtDoc

        txtDoc = [txtDoc]
        changeVec = CountVectorizer(stop_words='english', vocabulary=self.vocabulary)
        doc = changeVec.fit_transform(txtDoc).toarray()
        print(doc.shape)
        logPrcdi = np.sum((doc * self.logPrwc), axis=1).reshape((8,1)) + self.logPrc
        predIndex = np.argmax(logPrcdi)
        pred = classLabels[predIndex]
        Prcdi = np.exp(logPrcdi)
        prob = Prcdi[predIndex] / np.sum(Prcdi) * 100
        prob = round(float(prob),4)

        Prwc = np.exp(self.logPrwc[predIndex])
        words_with_prob = dict()#dict(zip(self.vocabulary, Prwc))
        for word, index in self.vocabulary.items():
            if doc[0][index] > 0:
                words_with_prob[word] = Prwc[index]
        
        num_of_words = max(len(words_with_prob), 10)
        res = dict(sorted(words_with_prob.items(), key = itemgetter(1), reverse = True)[:num_of_words])
        print(res)
        words = list(res.keys())
        dist = list(res.values())
        return {"Label":pred, "Probability":prob, "Words":words, "Distribution":dist}

    def add_word_vocabulary(self, word):
        index = len(self.vocabulary)
        if (self.vocabulary.get(word) == None):
            self.vocabulary[word] = index
            self.word_weight = np.append(self.word_weight, 1)
        print(self.vocabulary)
    
    def rem_word_vocabulary(self, word):
        if (self.vocabulary.get(word) != None):
            index = self.vocabulary.pop(word)
            self.word_weight = np.delete(self.word_weight, index)
            for j in range(index+1,len(self.vocabulary)+1):
                for key, value in self.vocabulary.items():
                    if value == j:
                        self.vocabulary[key] = j-1
        print(self.vocabulary)

    def adj_weight(self, word, weight):
        if (self.vocabulary.get(word) != None):
            index = self.vocabulary[word]
            self.word_weight[index] = weight