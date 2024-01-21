import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import sys

sys.path.append("c:/Workspace/AIKONG/Playground/Playground/experiment/keras/")
from custom_file_name import csv_file_name

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

path = 'c:/Workspace/AIKONG/_data/dacon/dechul/'

# 데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
y = train_csv["대출등급"]
ohe_y = ohe.fit_transform(y.values.reshape(-1, 1))

lbe = LabelEncoder()
#주택소유상태
train_csv["주택소유상태"] = lbe.fit_transform(train_csv["주택소유상태"])
test_csv["주택소유상태"] = lbe.transform(test_csv["주택소유상태"])
#대출목적
train_csv["대출목적"] = lbe.fit_transform(train_csv["대출목적"])
if '결혼' not in lbe.classes_:
    lbe.classes_ = np.append(lbe.classes_, '결혼')
test_csv["대출목적"] = lbe.transform(test_csv["대출목적"])
# 근로기간
train_csv["근로기간"] = lbe.fit_transform(train_csv["근로기간"])
test_csv["근로기간"] = lbe.transform(test_csv["근로기간"])
# 대출기간
train_csv["대출기간"] = lbe.fit_transform(train_csv["대출기간"])
test_csv["대출기간"] = lbe.transform(test_csv["대출기간"])

#대출등급 - 마지막 Label fit
train_csv["대출등급"] = lbe.fit_transform(train_csv["대출등급"])

x = train_csv.drop("대출등급", axis=1)
y = train_csv["대출등급"]

model = load_model("C:\Workspace\AIKONG\_high_score_file/0.92646\dacon_dechul_7191-0.184543.hdf5")

# 평가, 예측
submission = np.argmax(model.predict(test_csv), axis=1)
submission = lbe.inverse_transform(submission)

submission_csv["대출등급"] = submission

file_name = csv_file_name('sampleSubmission')
file_path = path + file_name
submission_csv.to_csv(file_path, index=False)