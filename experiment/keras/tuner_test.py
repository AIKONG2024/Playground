# 적절한 모델을 찾아주는 Tuner 사용해보기 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#1. 데이터 - 경로데이터를 메모리에 땡겨옴
path = "/Users/kongseon-eui/Documents/Workspace/AI_Project/_data/dacon/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.fillna(test_csv.mean()) # 715 non-null
test_csv = test_csv.fillna(test_csv.mean()) # 715 non-null
x = train_csv.drop(['count'], axis=1) #axis 0이 행 1이 열
y = train_csv['count'] 
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.8, random_state=12345)



from keras_custom_pk_pk.hyper_model import LeanerRegressionModel
build_model = LeanerRegressionModel(num_classes= 10, output_count=1)   
from keras.callbacks import EarlyStopping
es = [EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights= True)]
# =========RandomSearch 로 찾는법
# from keras_tuner.tuners import RandomSearch
# tuner = RandomSearch(
#     build_model,
#     objective='val_mae',
#     max_trials=100, # 이거는 랜덤으로 100번까지 찾는 방법
#     executions_per_trial=3,
#     directory='/Users/kongseon-eui/Documents/Workspace/AI_Project/_data/',
#     project_name='Keras Tuner Test')


# tuner.search(x_train, y_train, epochs=300, batch_size=1000, validation_split=0.2, callbacks = [es])
# tuner.search_space_summary()
# tuner.results_summary()

# #check results
# model = tuner.get_best_models(num_models = 1)[0]
# 

#============= Hyperband로 찾는법
from keras_tuner.tuners import Hyperband
tuner = Hyperband(build_model, objective='val_loss', max_epochs=1000, factor=3, 
                  directory = '/Users/kongseon-eui/Documents/Workspace/AI_Project/_data/', project_name = 'hyperband')
tuner.search(x_train, y_train, epochs = 30, validation_split = 0.2)

#get hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=3)[0]
tuner.results_summary()
model = tuner.hypermodel.build(best_hps)

history=model.fit(x_train, y_train, epochs =300, batch_size = 1000, validation_split = 0.2,  callbacks = [es])
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
submit = model.predict(test_csv)
submission_csv['count'] = submit

#제출
######### submission.csv 만들기(count컬럼에 값만 넣어주면됨) ############
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
file_path = path + f"submission_{save_time}.csv"
submission_csv.to_csv(file_path, index=False)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(history.history['val_loss'], color = 'red', label ='val_loss', marker='.')
plt.title("ddarung")
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()
