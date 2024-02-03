
#  https://dacon.io/competitions/open/235610/mysubmission
import sys
sys.path.append("/Users/kongseon-eui/Documents/Workspace/AI_Project/modules")
from keras_custom_pk_pk.hyper_model import MulticlassClassificationModel

from keras_tuner.tuners import Hyperband
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from keras.callbacks import EarlyStopping

#early stopping 
es = [EarlyStopping(monitor='val_loss', mode='min', patience=1000, restore_best_weights= True)]

#데이터 가져오기
path = '/Users/kongseon-eui/Documents/Workspace/AI_Project/_data/dacon/wine/'
train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

#전처리
train_csv.drop(columns='fixed acidity', inplace=True)
test_csv.drop(columns='fixed acidity', inplace=True)
train_csv.drop(columns='alcohol', inplace=True)
test_csv.drop(columns='alcohol', inplace=True)
train_csv.drop(columns='density', inplace=True)
test_csv.drop(columns='density', inplace=True)
train_csv.drop(columns='free sulfur dioxide', inplace=True)
test_csv.drop(columns='free sulfur dioxide', inplace=True)
train_csv.drop(columns='residual sugar', inplace=True)
test_csv.drop(columns='residual sugar', inplace=True)

#글자 제거
train_csv['type'] = train_csv['type'].replace({'white':1, 'red': 0})
test_csv['type'] = test_csv['type'].replace({'white':1, 'red': 0})


x = train_csv.drop(columns='quality')
y = train_csv['quality']


#one hot
y = y.values.reshape((-1,1))
one_hot_y = OneHotEncoder(sparse=False).fit_transform(y)
unique, count =  np.unique(one_hot_y, return_counts=True)
print(one_hot_y)

x_train, x_test, y_train, y_test = train_test_split(x, one_hot_y, train_size=0.85, random_state=123456, stratify= one_hot_y)

#Hyper Model
build_model = MulticlassClassificationModel(num_classes = 0, output_count = 7)
tuner = Hyperband(build_model, objective='val_loss', max_epochs=500, factor=3,
                  directory = path, project_name='hyperband', max_consecutive_failed_trials = 14, max_retries_per_trial = 14) 
tuner.search(x_train, y_train, epochs = 500, batch_size = 1000, validation_split = 0.2, callbacks = [es])

#get hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
tuner.results_summary()
model = tuner.hypermodel.build(best_hps)

history = model.fit(x_train, y_train, epochs = 500, batch_size = 1000, validation_split = 0.2, callbacks = [es])
submit = np.argmax(model.predict(test_csv), axis=1) + 3
print(pd.value_counts(submit))
submission_csv['quality'] = submit
'''
6    582
7    270
5    134
4      9
3      3
8      2

6    511
5    436
7     45
4      6
3      2
'''
#제출
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
file_path = path + f"submission_{save_time}.csv"
submission_csv.to_csv(file_path, index=False)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(history.history['val_loss'], color = 'red', label ='val_loss', marker='.')
plt.plot(history.history['loss'], color = 'green', label ='loss', marker='.')
plt.plot(history.history['val_acc'], color = 'blue', label ='val_acc', marker='.')
plt.title("wine")
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()