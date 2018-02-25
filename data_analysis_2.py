import pandas as pd
from matplotlib import pyplot as plt
import nltk

dfTrain = pd.read_csv("train.csv")


# get training features and labels
x_true = dfTrain.loc[dfTrain['spoiler'] == 1]
x_false = dfTrain.loc[dfTrain['spoiler'] == 0]

true_sentences = list(x_true['sentence'])
false_sentences = list(x_false['sentence'])

true = {'WP': 0, 'IN': 0, 'EX': 0, 'PRP': 0, 'CC': 0, 'VBZ': 0}
false = {'WP': 0, 'IN': 0, 'EX': 0, 'PRP': 0, 'CC': 0, 'VBZ': 0}

for sentence in true_sentences:
    text = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(text)
    for tag in tags:
        if tag[-1] == 'WP':
            true['WP'] += 1
        if tag[-1] == 'IN':
            true['IN'] += 1
        if tag[-1] == 'EX':
            true['EX'] += 1
        if tag[-1] == 'PRP':
            true['PRP'] += 1
        if tag[-1] == 'CC':
            true['CC'] += 1
        if tag[-1] == 'VBZ':
            true['VBZ'] += 1

for sentence in false_sentences:
    text = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(text)
    for tag in tags:
        if tag[-1] == 'WP':
            false['WP'] += 1
        if tag[-1] == 'IN':
            false['IN'] += 1
        if tag[-1] == 'EX':
            false['EX'] += 1
        if tag[-1] == 'PRP':
            false['PRP'] += 1
        if tag[-1] == 'CC':
            false['CC'] += 1
        if tag[-1] == 'VBZ':
            false['VBZ'] += 1


plt.bar(true.keys(), true.values())
plt.bar(false.keys(), false.values())
plt.show()


