import pandas as pd
import csv
import re
import numpy as np


def genre_feature(examples):
    genre = pd.read_csv("genre.csv", names=['series', 'genre'])
    genre_vec = ['unknown']*len(examples)
    for idx, series in enumerate(examples):
        print(idx)
        k = genre.loc[genre['series'] == series]
        try:
            if k['genre'].values:
                genre_vec[idx] = re.sub(r'[^a-zA-Z ]+', '', k['genre'].values[0])
        except ValueError:
            genre_vec[idx] = 'unknown'
        except TypeError:
            genre_vec[idx] = 'unknown'
    return genre_vec







