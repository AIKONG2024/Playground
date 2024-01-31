import pandas as pd
import numpy as np

path = "C:\_data\kaggle\jena\\"
jena_csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col="Date Time")
time_steps = 10
# jena_csv.iloc[:, -1]


def split_xy(dataFrame, size):
    xt, yt = [], []
    for i in range(len(dataFrame) - time_steps):
        x = dataFrame.iloc[i : i + size, :-1]
        y = dataFrame.iloc[i:-1]
        xt.append(x)
        yt.append(y)
    return xt, yt


x, y = split_xy(jena_csv, time_steps)
print(x)
