import pandas as pd
import re

# data_frame = pd.read_csv('series_data.csv',  encoding="ISO-8859-1")
#
# titles_1 = pd.read_csv("train.csv")
# titles_2 = pd.read_csv("test.csv")
#
# list_1 = list(set(list(titles_1["page"])))
# list_2 = list(set(list(titles_2["page"])))
# list_1 = list_1 + list_2
# print(len(list_1))
# present = pd.DataFrame()
# count = 0
#
# for series in list_1:
#     print(series)
#     if series:
#         k = re.sub(r"\B([A-Z])", r" \1", series)
#         true_frame = data_frame.loc[data_frame['primaryTitle'] == k]
#         if ~true_frame.empty:
#             present = present.append(true_frame)
#             count += 1
#
#
# present.to_csv('data_available.csv')
# print(count)







# print(data_frame.titleType.unique())
# x_true = data_frame.loc[data_frame['titleType'].isin(['tvseries','tvMiniSeries','tvEpisode'])]
# print(x_true)
# len(x_true)
# x_true.to_csv('series_data.csv')
