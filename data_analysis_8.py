import pandas as pd
import csv
import re
import numpy as np


def seasons_feature(examples):
    seasons = pd.read_csv("seasons.csv", names=['series', 'seasons'])
    seasons_vec = [0]*len(examples)
    for idx, series in enumerate(examples):
        print(idx)
        k = seasons.loc[seasons['series'] == series]
        try:
            print(k['seasons'])
            if k['seasons'].values >= 0:
                seasons_vec[idx] = k['seasons'].values[0]
        except ValueError:
            seasons_vec[idx] = 1
        except TypeError:
            seasons_vec[idx] = 1

    return seasons_vec







