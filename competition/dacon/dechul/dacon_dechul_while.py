
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras_tuner.tuners import Hyperband
import sys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time

# custom 모듈 import
sys.path.append('C:/MyPackages/')
from keras_custom_pk.hyper_model import MulticlassClassificationModel
from keras_custom_pk.file_name import *
from keras_custom_pk.callbacks import CustomEarlyStoppingAtLoss

path = "C:/_data/dacon/dechul/"
# 데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

# print(train_csv.head(25))


def splits(s):
    return int(s.split()[0])


def extract_근로기간(s):
    if "10+ " in s or "10+" in s:
        return 11  # 격차를 최대한 작게 1단위
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


train_csv["대출기간"] = train_csv["대출기간"].apply(splits)
test_csv["대출기간"] = test_csv["대출기간"].apply(splits)

train_csv["근로기간"] = train_csv["근로기간"].apply(extract_근로기간)
test_csv["근로기간"] = test_csv["근로기간"].apply(extract_근로기간)

# ============================레이블 인코딩==============================
# 레이블 인코딩
lbe = LabelEncoder()

# 주택소유상태
train_csv = train_csv[train_csv["주택소유상태"] != "ANY"] 
train_csv["주택소유상태"] = lbe.fit_transform(train_csv["주택소유상태"])
test_csv["주택소유상태"] = lbe.transform(test_csv["주택소유상태"])

# 대출목적
train_csv["대출목적"] = lbe.fit_transform(train_csv["대출목적"])
lbe.classes_ = np.append(lbe.classes_, "결혼")
lbe.classes_ = np.append(lbe.classes_, "기타")


# 대출등급 - 마지막 Label fit
train_csv["대출등급"] = lbe.fit_transform(train_csv["대출등급"])

# ===============X, y 분리=============================
x = train_csv.drop("대출등급", axis=1)
y = train_csv["대출등급"]

from sklearn.preprocessing import OneHotEncoder

# randbatch = np.random.randint(100, 1200)
randbatch = 1000
# rand_state = np.random.randint(777, 7777)
rand_state = 1234

# 데이터 분류
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.80, random_state=rand_state, stratify=y
)
print(np.unique(y_test, return_counts=True))

from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    StandardScaler,
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


build_model = MulticlassClassificationModel(num_classes=0, output_count=7)

tuner = Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=1000,
    factor=3,
    executions_per_trial=1,
    directory="C:\_data\dacon\dechul\\",
    project_name="hyperband",
)

tuner.search(
    x_train,
    y_train,
    epochs=100000,
    batch_size=1000,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=1000, restore_best_weights=True)],
)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
tuner.results_summary()
model = tuner.hypermodel.build(best_hps)

while 1 :
    history = model.fit(
        x_train,
        y_train,
        epochs=100000,
        batch_size=randbatch,
        verbose=0,
        validation_split=0.2,
        callbacks=[
            CustomEarlyStoppingAtLoss(patience=2000, monitor='val_loss', overfitting_stop_line=1.0, overfitting_count = 30, is_log = True)
            # EarlyStopping(monitor="val_loss", mode="min", patience=3000, restore_best_weights=True)
            ],
    )

    # 평가, 예측
    loss = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)
    arg_y_predict = np.argmax(y_predict, axis=1)
    f1 = f1_score(y_test, arg_y_predict, average="macro")

    # ==========================confirm score========================================
    print("=" * 100)
    print("loss : ", loss)
    print("f1_score :", f1)
    print("=" * 100)


    # ============================store Data===========================================
    submission = np.argmax(model.predict(test_csv), axis=1)
    submission = lbe.inverse_transform(submission)
    submission_csv["대출등급"] = submission
    file_name = csv_file_name(
        path, f"sampleSubmission_loss_{loss[0]:04f}_f1_{f1:04f}_"
    )
    submission_csv.to_csv(file_name, index=False)
    h5_file_name_d = h5_file_name(path, f"dechulModel_loss_{loss[0]:04f}_f1_{f1:04f}_batch_{randbatch}_rans_state_{rand_state}_")
    model.save(h5_file_name_d)

    # =============================visualization=======================================
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(9, 6))
    # plt.plot(history.history["val_loss"], color="red", label="val_loss", marker=".")
    # plt.plot(history.history["val_acc"], color="blue", label="val_acc", marker=".")
    # plt.xlabel = "epochs"
    # plt.ylabel = "loss"
    # plt.show()
