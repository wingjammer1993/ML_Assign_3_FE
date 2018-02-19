import pandas as pd
from matplotlib import pyplot as plt
import nltk

dfTrain = pd.read_csv("train.csv")


# get training features and labels
x_true = dfTrain.loc[dfTrain['spoiler'] == 1]
x_false = dfTrain.loc[dfTrain['spoiler'] == 0]

true_sentences = list(x_true['sentence'])
false_sentences = list(x_false['sentence'])

print(len(x_true.page.unique()))
print(len(x_false.page.unique()))
print(len(dfTrain.page.unique()))

