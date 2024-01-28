import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sys
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# custom 모듈 import
sys.path.append("c://Playground/experiment/keras/")
from custom_file_name import csv_file_name, h5_file_name

#1. 데이터
# bring data
path = "c://_data/dacon/dechul/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(train_csv.columns)

'''
Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
       '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수', '대출등급'],
      dtype='object')
'''

print(train_csv['연간소득'].value_counts())
print(test_csv['연간소득'].value_counts())


# Preprocessing
# ============================수치화===============================
def splits(s):
    return int(s.split()[0])   

def extract_근로기간(s):
    if "10+ " in s or "10+" in s:
        return 11. #격차를 최대한 작게 1단위
    elif "< 1" in s or "<1" in s:
        return 0.1
    elif "Unknown" in s:
        return 0.
    elif "1 years" in s:
        return 1.
    elif "3" == s:
        return 3.
    else:
        return splits(s)
train_csv["대출기간"] = train_csv["대출기간"].apply(splits)
test_csv["대출기간"] = test_csv["대출기간"].apply(splits)

train_csv["근로기간"] = train_csv["근로기간"].apply(extract_근로기간)
test_csv["근로기간"] = test_csv["근로기간"].apply(extract_근로기간)

# train_csv = train_csv["주택소유상태"] != 'ANY'

# train_csv["대출기간"] = train_csv["대출기간"].apply(splits)
# test_csv["대출기간"] = test_csv["대출기간"].apply(splits)

# train_csv["근로기간"] = train_csv["근로기간"].apply(extract_근로기간)
# test_csv["근로기간"] = test_csv["근로기간"].apply(extract_근로기간)

# value_counts = train_csv['대출금액'].value_counts()
# to_remove = value_counts[value_counts < 100].index
# train_csv = train_csv[~train_csv['대출금액'].isin(to_remove)]

# value_counts = train_csv['연간소득'].value_counts()
# to_remove = value_counts[value_counts < 100].index
# train_csv = train_csv[~train_csv['연간소득'].isin(to_remove)]

# value_counts = train_csv['총상환원금'].value_counts()
# to_remove = value_counts[value_counts < 100].index
# train_csv = train_csv[~train_csv['총상환원금'].isin(to_remove)]

# value_counts = train_csv['총상환이자'].value_counts()
# to_remove = value_counts[value_counts < 100].index
# train_csv = train_csv[~train_csv['총상환이자'].isin(to_remove)]

# value_counts = train_csv['연체계좌수'].value_counts()
# to_remove = value_counts[value_counts < 24].index
# train_csv = train_csv[~train_csv['연체계좌수'].isin(to_remove)]

#낮은 빈도 데이터 삭제
# train_csv.drop(train_csv[train_csv['대출등급'] == 'G'].index, inplace= True)
# train_csv.drop(train_csv[train_csv['대출등급'] == 'F'].index, inplace= True)

#원핫처리 (Data Leakage 방지)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe_train_df = pd.DataFrame(ohe.fit_transform(train_csv['대출목적'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['대출목적']))
train_csv = pd.concat([train_csv.reset_index(drop=True), ohe_train_df.reset_index(drop=True)], axis=1)
train_csv.drop('대출목적', axis=1, inplace=True)
ohe_test_df = pd.DataFrame(ohe.transform(test_csv['대출목적'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['대출목적']))
test_csv = pd.concat([test_csv.reset_index(drop=True), ohe_test_df.reset_index(drop=True)], axis=1)
test_csv.drop('대출목적', axis=1, inplace=True)

# 레이블 인코딩
lbe = LabelEncoder()
# 주택소유상태
train_csv["주택소유상태"] = lbe.fit_transform(train_csv["주택소유상태"])
test_csv["주택소유상태"] = lbe.transform(test_csv["주택소유상태"])

# 대출목적
# lbe.classes_ = np.union1d(train_csv["대출목적"].unique(), test_csv["대출목적"].unique())
# train_csv["대출목적"] = lbe.transform(train_csv["대출목적"])
# test_csv["대출목적"] = lbe.transform(test_csv["대출목적"])

# 근로기간
# train_csv["근로기간"] = lbe.fit_transform(train_csv["근로기간"])
# test_csv["근로기간"] = lbe.transform(test_csv["근로기간"])

# # 대출기간
# train_csv["대출기간"] = lbe.fit_transform(train_csv["대출기간"])
# test_csv["대출기간"] = lbe.transform(test_csv["대출기간"])

# 대출등급 - 마지막 Label fit
train_csv["대출등급"] = lbe.fit_transform(train_csv["대출등급"])

#===============X, y 분리=============================
X = train_csv.drop("대출등급", axis=1)
y = train_csv["대출등급"]

#===============이상치 제거=============================
from imblearn.over_sampling import SMOTE

print(y.value_counts())

smote = SMOTE(
    random_state=777,
    sampling_strategy={
        0: 16772,
        1: 28817,
        2: 27623,
        3: 13354,
        4: 7354,
        5: 1954,
        6: 420
    },
     k_neighbors=3
)


value_counts = X['대출금액'].value_counts()
to_remove = value_counts[value_counts < 50].index
train_csv = X[~X['대출금액'].isin(to_remove)]
# X, y = smote.fit_resample(X, y)

# value_counts = X['근로기간'].value_counts()
# to_remove = value_counts[value_counts < 50].index
# train_csv = X[~X['근로기간'].isin(to_remove)]
# # X, y = smote.fit_resample(X, y)

# value_counts = X['주택소유상태'].value_counts()
# to_remove = value_counts[value_counts < 50].index
# train_csv = X[~X['주택소유상태'].isin(to_remove)]
# # X, y = smote.fit_resample(X, y)

X, y = smote.fit_resample(X, y)

value_counts = X['연간소득'].value_counts()
to_remove = value_counts[value_counts < 50].index
train_csv = X[~X['연간소득'].isin(to_remove)]
# X, y = smote.fit_resample(X, y)

value_counts = X['총상환원금'].value_counts()
to_remove = value_counts[value_counts < 50].index
train_csv = X[~X['총상환원금'].isin(to_remove)]
# X, y = smote.fit_resample(X, y)

value_counts = X['최근_2년간_연체_횟수'].value_counts()
to_remove = value_counts[value_counts < 50].index
train_csv = X[~X['최근_2년간_연체_횟수'].isin(to_remove)]
# X, y = smote.fit_resample(X, y)

X, y = smote.fit_resample(X, y)

value_counts = X['총상환원금'].value_counts()
to_remove = value_counts[value_counts < 50].index
train_csv = X[~X['총상환원금'].isin(to_remove)]
# X, y = smote.fit_resample(X, y)

value_counts = X['부채_대비_소득_비율'].value_counts()
to_remove = value_counts[value_counts < 50].index
train_csv = X[~X['부채_대비_소득_비율'].isin(to_remove)]
# X, y = smote.fit_resample(X, y)

value_counts = X['총계좌수'].value_counts()
to_remove = value_counts[value_counts < 50].index
train_csv = X[~X['총계좌수'].isin(to_remove)]
# X, y = smote.fit_resample(X, y)

value_counts = X['총상환이자'].value_counts()
to_remove = value_counts[value_counts < 100].index
train_csv = X[~X['총상환이자'].isin(to_remove)]
# X, y = smote.fit_resample(X, y)

value_counts = X['연체계좌수'].value_counts()
to_remove = value_counts[value_counts < 24].index
train_csv = X[~X['연체계좌수'].isin(to_remove)]

X, y = smote.fit_resample(X, y)

print(train_csv.shape)
