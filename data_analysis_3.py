import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

dfTrain = pd.read_csv("train.csv")

# get training features and labels
x_true = dfTrain.loc[dfTrain['spoiler'] == 1]
x_false = dfTrain.loc[dfTrain['spoiler'] == 0]

true_sentences = list(x_true['sentence'])
false_sentences = list(x_false['sentence'])

true = {}
sid = SentimentIntensityAnalyzer()
for sentence in true_sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in ss:
        print('{0}: {1}, '.format(k, ss[k]), end='')
        if k in true:
            true[k] +=1
        else:
            true[k] = 1
print(true)
