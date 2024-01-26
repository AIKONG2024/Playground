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

sys.path.append("c:/Workspace/AIKONG/Playground/Playground/experiment/keras/")
# sys.path.append("c:/Playground/experiment/keras/")#102
# sys.path.append("c:/Playground/Playground/experiment/keras/") #122


from custom_file_name import csv_file_name, h5_file_name

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

path = 'c:/Workspace/AIKONG/_data/dacon/dechul/'
# path = 'c:/_data/dacon/dechul/' #102, 122


# 데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

# 데이터 전처리

# Onehot

lbe = LabelEncoder()
#주택소유상태
test_csv["주택소유상태"] = lbe.fit_transform(test_csv["주택소유상태"])
train_csv["주택소유상태"] = lbe.fit_transform(train_csv["주택소유상태"])
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

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
y = y.values.reshape(-1, 1)
ohe_y = ohe.fit_transform(y)

# 데이터 분류
x_train, x_test, y_train, y_test = train_test_split(
    x, ohe_y, train_size=0.85, random_state=1234567, stratify=ohe_y
)
print(np.unique(y_test, return_counts=True))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# print(x_train.shape) 
# print(y_train.shape)  

# 모델 생성
model = Sequential()
model.add(Dense(64, input_shape=(len(x.columns),)))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(128, activation='swish'))
model.add(Dropout(0.15))
model.add(Dense(32, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(7, activation="softmax"))

es = EarlyStopping(
    monitor="val_loss", mode="min", patience=1000, restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
    filepath="..\_data\_save\MCP\dacon_dechul_{epoch:02d}-{val_loss:2f}.hdf5",
    initial_value_threshold=0.26
)

# 컴파일 , 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
history = model.fit(
    x_train,
    y_train,
    epochs=100000,
    batch_size=1000,
    verbose=1,
    validation_split=0.2,
    callbacks=[es, mcp],
)

# 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스값 : ", loss)
y_predict = model.predict(x_test)
arg_y_test = np.argmax(y_test, axis=1)
arg_y_predict = np.argmax(y_predict, axis=1)

f1_score = f1_score(arg_y_test, arg_y_predict, average="macro")
print("f1_score :", f1_score)
submission = np.argmax(model.predict(test_csv), axis=1)
submission = lbe.inverse_transform(submission)

submission_csv["대출등급"] = submission

file_name = csv_file_name(path, f'sampleSubmission_f1_{f1_score:04f}_')
submission_csv.to_csv(file_name, index=False)
h5_file_name = h5_file_name(path, f'dechulModel_f1_{f1_score:04f}_')
model.save(h5_file_name)


import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
plt.plot(history.history["val_loss"], color="red", label="val_loss", marker=".")
plt.plot(history.history["val_acc"], color="blue", label="val_acc", marker=".")
plt.xlabel = "epochs"
plt.ylabel = "loss"
plt.show()