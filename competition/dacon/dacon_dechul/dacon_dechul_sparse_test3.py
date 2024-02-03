import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import sys
from keras.optimizers import Adam

sys.path.append("c://Playground/")
# sys.path.append("c:/Playground/Playground/") #122
# sys.path.append("c:/Playground/")#102


from custom_pk.custom_file_name import csv_file_name, h5_file_name

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

path = "c://_data/dacon/dechul/"

# path = 'c:/_data/dacon/dechul/' #102, 122


# 데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

# 데이터 전처리

"""
===============범주형 데이터 전처리 방식================
원핫 :주택소유상태, 대출목적  
라벨 :근로기간, 근무기간 
(근로기간은 이상한 데이터 수정)


자기전에 성찰
근로기간 변경?? 필요할것 같음.
근로기간이 핵심 포인트.
원핫으로도 처리해보기

====점검
대출기간: 레이블


전부 원핫으로 처리해보면?? , RobustScaler ==> val_loss 0.189
전부 레이블로 처리하면? => evaluate_loss :0.182



======================================================
"""
# 근로기간 이상치 제거
# 테스트 파일에도 존재하기 때문에 변경하지 않음. (Data Leakage).
# train_csv['근로기간'] = train_csv['근로기간'].replace('<1 year', '< 1 year')
# train_csv['근로기간'] = train_csv['근로기간'].replace('3', '3 years')
# train_csv['근로기간'] = train_csv['근로기간'].replace('1 years', '1 year')
# test_csv['근로기간'] = test_csv['근로기간'].replace('<1 year', '< 1 year')
# test_csv['근로기간'] = test_csv['근로기간'].replace('3', '3 years')
# test_csv['근로기간'] = test_csv['근로기간'].replace('1 years', '1 year')
# print(test_csv['근로기간'].value_counts())


# Onehot
# ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
# 주택 소유상태
# ohe_train_df = pd.DataFrame(ohe.fit_transform(train_csv['주택소유상태'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['주택소유상태']))
# train_csv = pd.concat([train_csv.reset_index(drop=True), ohe_train_df.reset_index(drop=True)], axis=1)
# train_csv.drop('주택소유상태', axis=1, inplace=True)
# ohe_test_df = pd.DataFrame(ohe.transform(test_csv['주택소유상태'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['주택소유상태']))
# test_csv = pd.concat([test_csv.reset_index(drop=True), ohe_test_df.reset_index(drop=True)], axis=1)
# test_csv.drop('주택소유상태', axis=1, inplace=True)

# #대출목적
# ohe_train_df = pd.DataFrame(ohe.fit_transform(train_csv['대출목적'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['대출목적']))
# train_csv = pd.concat([train_csv.reset_index(drop=True), ohe_train_df.reset_index(drop=True)], axis=1)
# train_csv.drop('대출목적', axis=1, inplace=True)
# ohe_test_df = pd.DataFrame(ohe.transform(test_csv['대출목적'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['대출목적']))
# test_csv = pd.concat([test_csv.reset_index(drop=True), ohe_test_df.reset_index(drop=True)], axis=1)
# test_csv.drop('대출목적', axis=1, inplace=True)

# #근로기간
# ohe_train_df = pd.DataFrame(ohe.fit_transform(train_csv['근로기간'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['근로기간']))
# train_csv = pd.concat([train_csv.reset_index(drop=True), ohe_train_df.reset_index(drop=True)], axis=1)
# train_csv.drop('근로기간', axis=1, inplace=True)
# ohe_test_df = pd.DataFrame(ohe.transform(test_csv['근로기간'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['근로기간']))
# test_csv = pd.concat([test_csv.reset_index(drop=True), ohe_test_df.reset_index(drop=True)], axis=1)
# test_csv.drop('근로기간', axis=1, inplace=True)

# #대출기간
# ohe_train_df = pd.DataFrame(ohe.fit_transform(train_csv['대출기간'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['대출기간']))
# train_csv = pd.concat([train_csv.reset_index(drop=True), ohe_train_df.reset_index(drop=True)], axis=1)
# train_csv.drop('대출기간', axis=1, inplace=True)
# ohe_test_df = pd.DataFrame(ohe.transform(test_csv['대출기간'].values.reshape(-1,1)), columns=ohe.get_feature_names_out(['대출기간']))
# test_csv = pd.concat([test_csv.reset_index(drop=True), ohe_test_df.reset_index(drop=True)], axis=1)
# test_csv.drop('대출기간', axis=1, inplace=True)


lbe = LabelEncoder()
# 주택소유상태
train_csv["주택소유상태"] = lbe.fit_transform(train_csv["주택소유상태"])
test_csv["주택소유상태"] = lbe.transform(test_csv["주택소유상태"])
# 대출목적
train_csv["대출목적"] = lbe.fit_transform(train_csv["대출목적"])
if "결혼" not in lbe.classes_:
    lbe.classes_ = np.append(lbe.classes_, "결혼")
test_csv["대출목적"] = lbe.transform(test_csv["대출목적"])
# 근로기간
train_csv["근로기간"] = lbe.fit_transform(train_csv["근로기간"])
test_csv["근로기간"] = lbe.transform(test_csv["근로기간"])
# 대출기간
train_csv["대출기간"] = lbe.fit_transform(train_csv["대출기간"])
test_csv["대출기간"] = lbe.transform(test_csv["대출기간"])

# 대출등급 - 마지막 Label fit
train_csv["대출등급"] = lbe.fit_transform(train_csv["대출등급"])

x = train_csv.drop("대출등급", axis=1)
y = train_csv["대출등급"]

from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC

# smote = SMOTE(random_state=777, sampling_strategy='not majority')
# x, y = smote.fit_resample(x, y)
# categorical_features = [x.columns.get_loc("대출등급")]
# smotenc = SMOTENC(categorical_features=categorical_features, random_state=777, sampling_strategy='auto', k_neighbors=1)
# x,y = smotenc.fit_resample(x,y)

smote = SMOTE(
    random_state=777,
    sampling_strategy={0: 28817, 2: 28817, 3: 28817, 4: 28817, 5: 28817, 6: 28817},
)
x, y = smote.fit_resample(x, y)

# 데이터 분류
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, random_state=777, stratify=y
)
print(np.unique(y_test, return_counts=True))

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# print(x_train.shape)
# print(y_train.shape)  dsds
"""
은닉 뉴런의 수는 입력 레이어의 크기와 출력 레이어의 크기 사이에 있어야 합니다.
은닉 뉴런의 수는 입력 레이어 크기의 2/3에 출력 레이어 크기를 더한 값이어야 합니다.
은닉 뉴런의 수는 입력 레이어 크기의 두 배보다 작아야 합니다

입력레이어크기 : 13
1. 뉴런 7~13
레이어 :6
2. 뉴런 : 6/3 + 7 = 9
레이어 :8
3. 뉴런 : 8/3 + 7 = 10
출력레이어크기 : 7
"""
# 입력 레이어, 출력 레이어 크기 정의
input_layer_size = 13
output_layer_size = 7

# 은닉 레이어 뉴런 수 계산
hidden_layer_size = int((2 / 3) * input_layer_size + output_layer_size)

# 은닉 레이어 뉴런 수가 입력 레이어 크기의 두 배보다 작은지 확인
if hidden_layer_size > 2 * input_layer_size:
    print("은닉 레이어의 뉴런 수가 너무 많습니다.")
else:
    print(f"은닉 레이어의 뉴런 수: {hidden_layer_size}")

# 모델 생성
model = Sequential()
model.add(Dense(hidden_layer_size, input_shape=(input_layer_size,), activation="swish"))
model.add(Dense(hidden_layer_size + 2, activation="swish"))
model.add(Dense(hidden_layer_size - 2, activation="swish"))
model.add(Dense(hidden_layer_size + 3, activation="swish"))
model.add(Dense(hidden_layer_size + 10, activation="swish"))
model.add(Dense(hidden_layer_size - 3, activation="swish"))
model.add(Dense(hidden_layer_size + 3, activation="swish"))
model.add(Dense(hidden_layer_size - 3, activation="swish"))
model.add(Dense(output_layer_size, activation="softmax"))

es = EarlyStopping(
    monitor="val_loss", mode="min", patience=5000, restore_best_weights=True
)

# 컴파일 , 훈련
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["acc"],
)
history = model.fit(
    x_train,
    y_train,
    epochs=500000,
    batch_size=10000,
    verbose=1,
    validation_split=0.2,
    callbacks=[es],
)

# 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스값 : ", loss)
y_predict = model.predict(x_test)
arg_y_predict = np.argmax(y_predict, axis=1)

f1_score = f1_score(y_test, arg_y_predict, average="macro")
print("f1_score :", f1_score)

submission = np.argmax(model.predict(test_csv), axis=1)
submission = lbe.inverse_transform(submission)

submission_csv["대출등급"] = submission

file_name = csv_file_name(path, f"sampleSubmission_loss_{loss[0]:04f}_")
submission_csv.to_csv(file_name, index=False)
h5_file_name = h5_file_name(path, f"dechulModel_loss_{loss[0]:04f}_f1_{f1_score:04f}_")
model.save(h5_file_name)
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
plt.plot(history.history["val_loss"], color="red", label="val_loss", marker=".")
plt.plot(history.history["val_acc"], color="blue", label="val_acc", marker=".")
plt.xlabel = "epochs"
plt.ylabel = "loss"
plt.show()
