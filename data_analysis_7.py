from imdbpie import Imdb
import pandas as pd
import csv
import re

titles_1 = pd.read_csv("train.csv")
titles_2 = pd.read_csv("test.csv")

list_1 = list(set(list(titles_1["page"])))
list_2 = list(set(list(titles_2["page"])))
list_1 = list_1 + list_2
print(len(list_1))

count = 0
genre_dict = {}
seasons_dict = {}
imdb = Imdb()
for series in list_1:
    count += 1
    print(count)
    if series:
        try:
            title = re.sub(r"\B([A-Z])", r" \1", series)
            k = imdb.search_for_title(title)
            b = k[0]['imdb_id']
            year = k[0]['year']
            try:
                genre = imdb.get_title_genres(b)['genres']
                genre_dict[series] = genre
            except LookupError:
                print('LookupError for {}', series)
                genre_dict[series] = 'unknown'
            try:
                eps = len(imdb.get_title_episodes(b)['seasons'])
                seasons_dict[series] = eps
            except LookupError:
                print('LookupError for {}', series)
                seasons_dict[series] = 1
        except LookupError:
            print('LookupError for {}', series)
        except TimeoutError:
            print('TimeoutError for {}', series)


with open('gen.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in genre_dict.items():
        writer.writerow([key, value])

with open('sea.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in seasons_dict.items():
        writer.writerow([key, value])



