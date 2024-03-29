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
sys.path.append("C:/MyPackages/")
from keras_custom_pk.file_name import csv_file_name, h5_file_name

# 1. 데이터
# ============================이터 가져오기===========================
path = "c://_data/dacon/dechul/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


# ============================이상치 제거==============================
def splits(s):
    return int(s.split()[0])


def extract_근로기간(s):
    if "10+ " in s or "10+" in s:
        return 11.0  # 격차를 최대한 작게 1단위
    elif "< 1" in s or "<1" in s:
        return 0.1
    elif "Unknown" in s:
        return 0.0
    elif "1 years" in s:
        return 1.0
    elif "3" == s:
        return 3.0
    else:
        return splits(s)

train_csv["근로기간"] = train_csv["근로기간"].apply(extract_근로기간)
test_csv["근로기간"] = test_csv["근로기간"].apply(extract_근로기간)

train_csv["대출기간"] = train_csv["대출기간"].apply(splits)
test_csv["대출기간"] = test_csv["대출기간"].apply(splits)

# 짤짤이 데이터 제거
value_counts = train_csv["대출금액"].value_counts()
to_remove = value_counts[value_counts < 100].index
train_csv = train_csv[~train_csv["대출금액"].isin(to_remove)]

value_counts = train_csv["연간소득"].value_counts()
to_remove = value_counts[value_counts < 100].index
train_csv = train_csv[~train_csv["연간소득"].isin(to_remove)]

value_counts = train_csv["총상환원금"].value_counts()
to_remove = value_counts[value_counts < 100].index
train_csv = train_csv[~train_csv["총상환원금"].isin(to_remove)]

value_counts = train_csv["총상환이자"].value_counts()
to_remove = value_counts[value_counts < 100].index
train_csv = train_csv[~train_csv["총상환이자"].isin(to_remove)]

value_counts = train_csv["연체계좌수"].value_counts()
to_remove = value_counts[value_counts < 24].index
train_csv = train_csv[~train_csv["연체계좌수"].isin(to_remove)]

# 원핫처리 (Data Leakage 방지)
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
ohe_train_df = pd.DataFrame(
    ohe.fit_transform(train_csv["대출목적"].values.reshape(-1, 1)),
    columns=ohe.get_feature_names_out(["대출목적"]),
)
train_csv = pd.concat(
    [train_csv.reset_index(drop=True), ohe_train_df.reset_index(drop=True)], axis=1
)
train_csv.drop("대출목적", axis=1, inplace=True)
ohe_test_df = pd.DataFrame(
    ohe.transform(test_csv["대출목적"].values.reshape(-1, 1)),
    columns=ohe.get_feature_names_out(["대출목적"]),
)
test_csv = pd.concat(
    [test_csv.reset_index(drop=True), ohe_test_df.reset_index(drop=True)], axis=1
)
test_csv.drop("대출목적", axis=1, inplace=True)

# ============================레이블 인코딩==========================
lbe = LabelEncoder()
# 주택소유상태
train_csv["주택소유상태"] = lbe.fit_transform(train_csv["주택소유상태"])
test_csv["주택소유상태"] = lbe.transform(test_csv["주택소유상태"])

# 대출등급 - 마지막 Label fit
train_csv["대출등급"] = lbe.fit_transform(train_csv["대출등급"])

# ============================X,y 분리 ==============================
X = train_csv.drop("대출등급", axis=1)
y = train_csv["대출등급"]

# ============================ 데이터 증폭 ====================================
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    random_state=777,
    sampling_strategy={
        0: 28817,
        1: 28817,
        2: 28817,
        3: 28817,
        4: 28817,
        5: 28817,
        6: 28817,
    },
)
X, y = smote.fit_resample(X, y)


# ============================train,test 분리================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=1234567, stratify=y
)

# ============================스케일링===========================================
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
)

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)
test_csv = scaler.transform(test_csv)


# =================================모델링=============================================
input_layer_size = 13
output_layer_size = 7
hidden_layer_size = int((2 / 3) * input_layer_size + output_layer_size)
if hidden_layer_size > 2 * input_layer_size:
    print("뉴런수 많음...................")
    exit(0)
else:
    print(f"+++++++++뉴런 수: {hidden_layer_size}+++++++++++++++")
# 2. 모델 구성
# model = Sequential()
# model.add(Dense(hidden_layer_size, input_shape=(input_layer_size,), activation="relu"))
# model.add(Dense(hidden_layer_size +2, activation="relu"))
# model.add(Dense(hidden_layer_size -1, activation="relu"))
# model.add(Dense(hidden_layer_size +3, activation="relu"))
# model.add(Dense(hidden_layer_size -2, activation="relu"))
# model.add(Dense(hidden_layer_size +10, activation="relu"))
# model.add(Dense(hidden_layer_size -5, activation="relu"))
# model.add(Dense(hidden_layer_size +8, activation="relu"))
# model.add(Dense(hidden_layer_size -4, activation="relu"))
# model.add(Dense(hidden_layer_size +3, activation="relu"))
# model.add(Dense(hidden_layer_size -1, activation="relu"))
# model.add(Dense(hidden_layer_size +2, activation="relu"))
# model.add(Dense(hidden_layer_size -5, activation="relu"))
# model.add(Dense(output_layer_size, activation="softmax"))

model = Sequential()
model.add(Dense(64, input_shape=(len(X.columns),)))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(7, activation="softmax"))


es = EarlyStopping(
    monitor="val_loss", mode="min", patience=1000, restore_best_weights=True
)

# 3. 컴파일 , 훈련
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["acc"],
)

history = model.fit(
    x_train,
    y_train,
    epochs=20000,
    batch_size=1000,
    verbose=1,
    validation_split=0.2,
    callbacks=[es],
)

# 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
arg_y_predict = np.argmax(y_predict, axis=1)
f1_score = f1_score(y_test, arg_y_predict, average="macro")

# ==========================score 확인========================================
print("=" * 100)
print("loss : ", loss)
print("f1_score :", f1_score)
print("=" * 100)


# ============================예측값 저장===========================================
submission = np.argmax(model.predict(test_csv), axis=1)
submission = lbe.inverse_transform(submission)
submission_csv["대출등급"] = submission
file_name = csv_file_name(
    path, f"sampleSubmission_loss_{loss[0]:04f}_f1_{f1_score:04f}_"
)
submission_csv.to_csv(file_name, index=False)
h5_file_name = h5_file_name(path, f"dechulModel_loss_{loss[0]:04f}_f1_{f1_score:04f}_")
model.save(h5_file_name)


# =============================시각화=======================================
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
plt.plot(history.history["val_loss"], color="red", label="val_loss", marker=".")
plt.plot(history.history["val_acc"], color="blue", label="val_acc", marker=".")
plt.xlabel = "epochs"
plt.ylabel = "loss"
plt.show()
