import pandas as pd
import re
dfTrain = pd.read_csv("train.csv")
genres = pd.read_csv("genre.csv", names=['series', 'genre'])

true_genres = {}
false_genres = {}
gen_vec = {}
count_true = 0
count_false = 0
tropes = dfTrain.trope.unique()
for index, row in dfTrain.iterrows():
    print(index)
    series = row['page']
    truth = row['spoiler']
    trope = row['trope']
    genre = genres.loc[genres['series'] == series]
    if truth:
        count_true += 1
        try:
            if genre['genre'].values:
                genress = (re.sub(r'[^a-zA-Z ]+', '', genre['genre'].values[0])).split(' ')
                for gen in genress:
                    if gen in true_genres:
                        if trope in true_genres[gen]:
                            true_genres[gen][trope] += 1
                        else:
                            true_genres[gen][trope] = 1
                    else:
                        true_genres[gen] = {}
                        true_genres[gen][trope] = 1
        except ValueError:
            print('lookup')
        except TypeError:
            print('lookup')
    else:
        count_false += 1
        try:
            if genre['genre'].values:
                genress = (re.sub(r'[^a-zA-Z ]+', '', genre['genre'].values[0])).split(' ')
                for gen in genress:
                    if gen in false_genres:
                        if trope in false_genres[gen]:
                            false_genres[gen][trope] += 1
                        else:
                            false_genres[gen][trope] = 1
                    else:
                        false_genres[gen] = {}
                        false_genres[gen][trope] = 1
        except ValueError:
            print('lookup')
        except TypeError:
            print('lookup')


for gen in true_genres:
    for trope in true_genres[gen]:
        if true_genres[gen][trope] > 5:
            print('true', gen, trope, true_genres[gen][trope])

for gen in false_genres:
    for trope in false_genres[gen]:
        if false_genres[gen][trope] > 5:
            print('false', gen, trope, false_genres[gen][trope])
print('done')
print(count_true)
print(count_false)












