import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

path = 'c:/Workspace/AIKONG/_data/dacon/dechul/'

# 데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

# 데이터 전처리

'''
===============범주형 데이터 전처리 방식================
원핫 :주택소유상태, 대출목적  
라벨 :근로기간, 근무기간 
(근로기간은 이상한 데이터 삭제)
======================================================
'''
#근로기간 이상치 제거
train_csv['근로기간'] = train_csv['근로기간'].replace('<1 year', '< 1 year')
train_csv['근로기간'] = train_csv['근로기간'].replace('3', '3 years')
train_csv['근로기간'] = train_csv['근로기간'].replace('1 years', '1 year')
test_csv['근로기간'] = test_csv['근로기간'].replace('<1 year', '< 1 year')
test_csv['근로기간'] = test_csv['근로기간'].replace('3', '3 years')
test_csv['근로기간'] = test_csv['근로기간'].replace('1 years', '1 year')
# print(test_csv['근로기간'].value_counts())


# Onehot
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
#주택 소유상태 
ohe_train_df = pd.DataFrame(ohe.fit_transform(train_csv['주택소유상태'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['주택소유상태']))
train_csv = pd.concat([train_csv.reset_index(drop=True), ohe_train_df.reset_index(drop=True)], axis=1)
train_csv.drop('주택소유상태', axis=1, inplace=True)
ohe_test_df = pd.DataFrame(ohe.transform(test_csv['주택소유상태'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['주택소유상태']))
test_csv = pd.concat([test_csv.reset_index(drop=True), ohe_test_df.reset_index(drop=True)], axis=1)
test_csv.drop('주택소유상태', axis=1, inplace=True)
#대출목적
ohe_train_df = pd.DataFrame(ohe.fit_transform(train_csv['대출목적'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['대출목적']))
train_csv = pd.concat([train_csv.reset_index(drop=True), ohe_train_df.reset_index(drop=True)], axis=1)
train_csv.drop('대출목적', axis=1, inplace=True)
ohe_test_df = pd.DataFrame(ohe.transform(test_csv['대출목적'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['대출목적']))
test_csv = pd.concat([test_csv.reset_index(drop=True), ohe_test_df.reset_index(drop=True)], axis=1)
test_csv.drop('대출목적', axis=1, inplace=True)

lbe = LabelEncoder()
#근로기간
test_csv["근로기간"] = lbe.fit_transform(test_csv["근로기간"])
train_csv["근로기간"] = lbe.fit_transform(train_csv["근로기간"])
#대출기간
test_csv["대출기간"] = lbe.fit_transform(test_csv["대출기간"])
train_csv["대출기간"] = lbe.fit_transform(train_csv["대출기간"])
#대출등급 - 마지막
train_csv["대출등급"] = lbe.fit_transform(train_csv["대출등급"])

x = train_csv.drop("대출등급", axis=1)
y = train_csv["대출등급"]
print(train_csv.shape)
