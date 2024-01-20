import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

path = 'c:/Workspace/AIKONG/_data/dacon/dechul/'

# 데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

# 데이터 전처리

one_hot_e = OneHotEncoder(sparse=False)
train_csv['근로기간'] = one_hot_e.fit_transform(train_csv['근로기간'].values.reshape(-1,1))
test_csv['근로기간'] = one_hot_e.fit_transform(test_csv['근로기간'].values.reshape(-1,1))
print(train_csv['근로기간'].value_counts())

train_le = LabelEncoder()
test_le = LabelEncoder()
train_csv["주택소유상태"] = train_le.fit_transform(train_csv["주택소유상태"])
train_csv["대출목적"] = train_le.fit_transform(train_csv["대출목적"])
# train_csv["근로기간"] = train_le.fit_transform(train_csv["근로기간"])
train_csv["대출기간"] = train_le.fit_transform(train_csv["대출기간"])
train_csv["대출등급"] = train_le.fit_transform(train_csv["대출등급"])

test_csv["주택소유상태"] = test_le.fit_transform(test_csv["주택소유상태"])
test_csv["대출목적"] = test_le.fit_transform(test_csv["대출목적"])
test_csv["대출기간"] = train_le.fit_transform(test_csv["대출기간"])

x = train_csv.drop("대출등급", axis=1)
y = train_csv["대출등급"]
# print(train_csv.head(30))
