import pandas as pd
from matplotlib import pyplot as plt

dfTrain = pd.read_csv("train.csv")


# get training features and labels
x_true = dfTrain.loc[dfTrain['spoiler'] == 1]
x_false = dfTrain.loc[dfTrain['spoiler'] == 0]

len_true = [len(x) for x in list(x_true["sentence"])]
len_false = [len(x) for x in list(x_false["sentence"])]

l1, = plt.plot(len_true, label='True')
l2, = plt.plot(len_false, label='False')
plt.title('Length distribution')
plt.legend(handles=[l1, l2])
plt.show()

grouped_trope_true = x_true.groupby('trope').size().sort_values(ascending=False)
grouped_trope_false = x_false.groupby('trope').size().sort_values(ascending=False)

plt.xticks(rotation='vertical')
plt.gca().set_xticklabels(grouped_trope_true.index)
l1, = plt.plot(grouped_trope_true[0:25].values, label='True')
l2, = plt.plot(grouped_trope_true[grouped_trope_false[0:25].index], label='False')
plt.legend(handles=[l1, l2])
plt.tight_layout()
plt.title('Top true trope distribution')
plt.show()

plt.gca().set_xticklabels(grouped_trope_false.index)
plt.legend()
plt.xticks(rotation='vertical')
l1, = plt.plot(grouped_trope_false[0:25].values, label='False')
l2, = plt.plot(grouped_trope_false[grouped_trope_true[0:25].index], label='True')
plt.legend(handles=[l1, l2])
plt.tight_layout()
plt.title('Top false trope distribution')
plt.show()


grouped_page_true = x_true.groupby('page').size().sort_values(ascending=False)
grouped_page_false = x_false.groupby('page').size().sort_values(ascending=False)

print(grouped_page_true[0:10])
print(grouped_page_false[0:10])

plt.xticks(rotation='vertical')
plt.gca().set_xticklabels(grouped_page_true.index)
l1, = plt.plot(grouped_page_true[0:10].values, label='True')
l2, = plt.plot(grouped_page_true[grouped_page_false[0:10].index], label='False')
plt.legend(handles=[l1, l2])
plt.tight_layout()
plt.title('Top true page distribution')
plt.show()

plt.gca().set_xticklabels(grouped_page_false.index)
plt.legend()
plt.xticks(rotation='vertical')
l1, = plt.plot(grouped_page_false[0:10].values, label='False')
l2, = plt.plot(grouped_page_false[grouped_page_true[0:10].index], label='True')
plt.legend(handles=[l1, l2])
plt.tight_layout()
plt.title('Top false page distribution')
plt.show()

