import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras_tuner.tuners import Hyperband
from custom_hyper_model import MulticlassClassificationModel

path = 'C:/_data/dacon/dechul/'
#데이터 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# print(train_csv.head(25))


#데이터 전처리
#1. 결측치 제거 '근로기간'
# train_csv = train_csv[train_csv['근로기간'] != 'Unknown']
# print(train_csv.head(50))

#2 문자 -> 수치 대상: 주택소유상태, 근로기간, 대출등급, 대출목적
unique, count = np.unique(train_csv['근로기간'], return_counts=True)
# print(unique, count)
train_le = LabelEncoder()
test_le = LabelEncoder()
train_csv['대출기간'] = train_le.fit_transform(train_csv['대출기간'])
train_csv['근로기간'] = train_le.fit_transform(train_csv['근로기간'])
train_csv['주택소유상태'] = train_le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = train_le.fit_transform(train_csv['대출목적'])

test_csv['대출기간'] = train_le.fit_transform(test_csv['대출기간'])
test_csv['근로기간'] = test_le.fit_transform(test_csv['근로기간'])
test_csv['주택소유상태'] = test_le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = test_le.fit_transform(test_csv['대출목적'])

#3. split 수치화 대상 int로 변경: 대출기간
# print(train_csv['대출기간'].str.split().str[0])
# train_csv['대출기간'] = train_csv['대출기간'].str.split().str[0].astype(int)
# test_csv['대출기간'] = test_csv['대출기간'].str.split().str[0].astype(int)
train_csv['대출등급'] = train_le.fit_transform(train_csv['대출등급'])


x = train_csv.drop('대출등급', axis=1)
y = train_csv['대출등급']

#결측치 확인
# print(x.isna().sum())
# print(y.isna().sum())

# print(x.head())

#클래스 확인
unique, count =  np.unique(y, return_counts=True)
print(unique , count)
print(x.shape)#(90623, 14)
print(y.shape)#(90623, 7)

from sklearn.preprocessing import OneHotEncoder
y = y.values.reshape(-1,1) 
one_hot_y = OneHotEncoder(sparse=False).fit_transform(y)

unique, count = np.unique(one_hot_y, return_counts=True)
print(unique, count) #[0. 1.] [543738  90623]

#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, one_hot_y, train_size=0.80, random_state=1234, stratify=one_hot_y)
print(np.unique(y_test, return_counts=True))

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape)#(77029, 13)
print(y_train.shape)#(77029, 7)
es = EarlyStopping(monitor='val_loss', mode = 'min', patience= 1000, restore_best_weights=True)
build_model = MulticlassClassificationModel(num_classes=0, output_count=7)

tuner = Hyperband(build_model, objective= 'val_loss', 
          max_epochs=30, 
          factor=3,
          directory = 'C:\_data\dacon\dechul/',
          project_name = 'hyperband',
          tune_new_entries=3,
          max_consecutive_failed_trials=5
          )

tuner.search(x_train, y_train, epochs = 100000, batch_size = 1000, validation_split = 0.2, callbacks = [es])
best_hps = tuner.get_best_hyperparameters(num_trials=3)[0]
tuner.results_summary()
model = tuner.hypermodel.build(best_hps)
es = EarlyStopping(monitor='val_loss', mode = 'min', patience= 1000, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=100000, batch_size=1000, verbose= 1, validation_split=0.2,
                    callbacks = [es])
'''
Search: Running Trial #250

Value             |Best Value So Far |Hyperparameter
9                 |5                 |num_layers
288               |368               |units_0
relu              |relu              |activation
192               |144               |units_1
0.01              |0.001             |learning_rate
192               |96                |units_2
336               |240               |units_3
288               |336               |units_4
496               |192               |units_5
240               |224               |units_6
256               |368               |units_7s
64                |64                |units_8
32                |400               |units_9
'''

# #모델 생성
# model = Sequential()
# model.add(Dense(368, input_shape = (13,)))
# model.add(Dense(144, activation='relu'))
# model.add(Dense(96))
# model.add(Dense(240))
# model.add(Dense(336))
# model.add(Dense(192))
# model.add(Dense(224))
# model.add(Dense(368))
# model.add(Dense(64))
# model.add(Dense(400))
# model.add(Dense(7, activation='softmax'))

# es = EarlyStopping(monitor='val_loss', mode = 'min', patience= 300, restore_best_weights=True)

# # 컴파일 , 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# history = model.fit(x_train, y_train, epochs=1000, batch_size=1000, verbose= 1, validation_split=0.2)

#평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스값 : ", loss)
y_predict = model.predict(x_test)
arg_y_test = np.argmax(y_test,axis=1)
arg_y_predict = np.argmax(y_predict, axis=1)

f1_score = f1_score(arg_y_test, arg_y_predict, average='macro') 
print("f1_score :", f1_score)
submission = np.argmax(model.predict(test_csv), axis=1)
submission = train_le.inverse_transform(submission)

submission_csv['대출등급'] = submission

import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
file_path = path + f"sampleSubmission{save_time}.csv"
submission_csv.to_csv(file_path, index=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(history.history['val_loss'], color = 'red', label ='val_loss', marker='.')
plt.plot(history.history['val_acc'], color = 'blue', label ='val_acc', marker='.')
plt.xlabel = 'epochs'
plt.ylabel = 'loss'
plt.show()




