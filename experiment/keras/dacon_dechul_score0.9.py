import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping , ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

path = 'C:/_data/dacon/dechul/'
#데이터 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# print(train_csv.head(25))


#데이터 전처리

#1 문자 -> 수치 대상: 주택소유상태, 근로기간, 대출등급, 대출목적
unique, count = np.unique(train_csv['근로기간'], return_counts=True)
print(unique, count)
unique, count = np.unique(test_csv['근로기간'], return_counts=True)
print(unique, count)
train_le = LabelEncoder()
test_le = LabelEncoder()
train_csv['주택소유상태'] = train_le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = train_le.fit_transform(train_csv['대출목적'])
train_csv['근로기간'] = train_le.fit_transform(train_csv['근로기간'])
train_csv['대출등급'] = train_le.fit_transform(train_csv['대출등급'])


test_csv['주택소유상태'] = test_le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = test_le.fit_transform(test_csv['대출목적'])
test_csv['근로기간'] = test_le.fit_transform(test_csv['근로기간'])

#2. 근로기간 따로 처리
# train_csv['근로기간'] = train_csv['근로기간'].replace({'1 year' : 1, '1 years': 1, '10+years': 10, '10+ years' : 10,
#                                               '2 years' : 2, '3': 3, '3 years' : 3, '4 years' : 4, '5 years' : 5,
#                                               '6 years' : 6, '7 years' : 7, '8 years' : 8, '9 years' : 9, '< 1 year' : 0.5, 
#                                               '<1 year' : 0.5, 'Unknown' : 0}).astype(float)
# test_csv['근로기간'] = test_csv['근로기간'].replace({'1 year' : 1, '1 years': 1, '10+years': 10, '10+ years' : 10,
#                                               '2 years' : 2, '3': 3, '3 years' : 3, '4 years' : 4, '5 years' : 5,
#                                               '6 years' : 6, '7 years' : 7, '8 years' : 8, '9 years' : 9, '< 1 year' : 0.5, 
#                                               '<1 year' : 0.5, 'Unknown' : 0}).astype(float)
# train_csv = train_csv[train_csv['근로기간'] != 'Unknown']
# print(test_csv.head(20))
# unique, count = np.unique(train_csv['근로기간'], return_counts=True)
# print(unique, count)
# unique, count = np.unique(test_csv['근로기간'], return_counts=True)
# print(unique, count)

#3. split 수치화 대상 int로 변경: 대출기간
print(train_csv['대출기간'].str.split().str[0])
train_csv['대출기간'] = train_csv['대출기간'].str.split().str[0].astype(float)
test_csv['대출기간'] = test_csv['대출기간'].str.split().str[0].astype(float)

x = train_csv.drop('대출등급', axis=1)
y = train_csv['대출등급']

#결측치 확인
print(x.isna().sum())
print(y.isna().sum())

print(x.head())

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
x_train, x_test, y_train, y_test = train_test_split(x, one_hot_y, train_size=0.85, random_state=1234567, stratify=one_hot_y)
print(np.unique(y_test, return_counts=True))

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape)#(77029, 13)
print(y_train.shape)#(77029, 7)

#모델 생성
model = Sequential()
model.add(Dense(16, input_shape = (13,)))
model.add(Dense(32,activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode = 'min', patience= 1000, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode= 'min', save_best_only=True, verbose=1, filepath='..\_data\_save\MCP\keras26_MCP_11_dacon_dechul.hdf5')

#컴파일 , 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10000, batch_size=1000, verbose= 1, validation_split=0.2, callbacks=[es, mcp])

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


'''
기존 : 
loss : [1.271729826927185, 0.5023883581161499]
============================
best : MaxAbs
============================
MinMaxScaler()
 - loss : [1.2414005994796753, 0.5350640416145325]
StandardScaler()
 - loss : [1.2407830953598022, 0.5290411710739136]
MaxAbsScaler()
 - loss :  [1.2420356273651123, 0.5340256094932556]
RobustScaler()
 - loss : [1.242053508758545, 0.522741436958313]
'''

