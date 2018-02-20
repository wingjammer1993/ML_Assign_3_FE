import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

dfTrain = pd.read_csv("train.csv")

# get training features and labels
x_true = dfTrain.loc[dfTrain['spoiler'] == 1]
x_false = dfTrain.loc[dfTrain['spoiler'] == 0]

true_sentences = list(x_true['sentence'])
false_sentences = list(x_false['sentence'])

true = {'pos': 0, 'neg': 0, 'neu': 0, 'compound': 0}
sid = SentimentIntensityAnalyzer()
for sentence in true_sentences:
    ss = sid.polarity_scores(sentence)
    if ss['pos'] > 0.5:
        true['pos'] += 1
    if ss['neg'] > 0.5:
        true['neg'] += 1
    if ss['neu'] > 0.5:
        true['neu'] += 1
    if -0.5 > ss['compound'] or ss['compound'] > 0.5:
        true['compound'] += 1

false = {'pos': 0, 'neg': 0, 'neu': 0, 'compound': 0}
sid = SentimentIntensityAnalyzer()
for sentence in false_sentences:
    ss = sid.polarity_scores(sentence)
    if ss['pos'] > 0.5:
        false['pos'] += 1
    if ss['neg'] > 0.5:
        false['neg'] += 1
    if ss['neu'] > 0.5:
        false['neu'] += 1
    if -0.5 > ss['compound'] or ss['compound'] > 0.5:
        false['compound'] += 1

print(true)
print(false)
