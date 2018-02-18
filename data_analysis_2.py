import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import nltk

dfTrain = pd.read_csv("train.csv")


# get training features and labels
x_true = dfTrain.loc[dfTrain['spoiler'] == 1]
x_false = dfTrain.loc[dfTrain['spoiler'] == 0]

true_sentences = list(x_true['sentence'])
false_sentences = list(x_false['sentence'])

true = {'VBD': 0, 'VBG': 0, 'VBN': 0, 'VBP': 0, 'VBZ': 0, 'UH': 0}
false = {'VBD': 0, 'VBG': 0, 'VBN': 0, 'VBP': 0, 'VBZ': 0, 'UH': 0}

for sentence in true_sentences:
    text = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(text)
    for tag in tags:
        if tag[-1] == 'VBD':
            true['VBD'] += 1
        if tag[-1] == 'VBG':
            true['VBG'] += 1
        if tag[-1] == 'VBN':
            true['VBN'] += 1
        if tag[-1] == 'VBP':
            true['VBP'] += 1
        if tag[-1] == 'VBZ':
            true['VBZ'] += 1
        if tag[-1] == 'UH':
            true['UH'] += 1

for sentence in false_sentences:
    text = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(text)
    for tag in tags:
        if tag[-1] == 'VBD':
            false['VBD'] += 1
        if tag[-1] == 'VBG':
            false['VBG'] += 1
        if tag[-1] == 'VBN':
            false['VBN'] += 1
        if tag[-1] == 'VBP':
            false['VBP'] += 1
        if tag[-1] == 'VBZ':
            false['VBZ'] += 1
        if tag[-1] == 'UH':
            false['UH'] += 1


plt.bar(true.keys(), true.values())
plt.bar(false.keys(), false.values())
plt.show()


