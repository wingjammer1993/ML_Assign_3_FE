import pandas as pd
import re
import numpy as np


def give_genre_vector(examples):
    data = pd.read_csv("data_available.csv")
    genres = []
    k = data.genres.unique()
    for idx, gen in enumerate(k):
        gen_list = gen.split(',')
        genres = genres + gen_list
    genres = list(set(genres))
    genres.remove('\\N')
    genre_vector = np.zeros((len(examples), len(genres)))
    print(examples)
    for index, series in enumerate(examples):
        if series:
            k = re.sub(r"\B([A-Z])", r" \1", series)
            true_frame = data.loc[data['primaryTitle'] == k]
            if len(true_frame.genres.unique()):
                gen_ls = []
                k_ls = true_frame.genres.unique()
                for idx, gen in enumerate(k_ls):
                    gen_list = gen.split(',')
                    gen_ls = gen_ls + gen_list
                genres_ls = set(gen_ls)
                if '\\N' in genres_ls:
                    genres_ls.remove('\\N')

                for gen in genres_ls:
                    idx = genres.index(gen)
                    genre_vector[index, idx] = 1

    return genre_vector



