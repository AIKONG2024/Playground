import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model, model_from_json, model_from_yaml
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

unique, counts = np.unique(y, return_counts=True) 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=1234
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# load model to json
with open('../_data/_save/save_test/json_model01.json', 'r') as f:
    model = model_from_json(f.read())

#load only model
# model = load_model("../_data/_save/save_test/model01.h5")

# 컴파일 , 훈련
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
# history = model.fit(x_train, y_train, epochs=76, batch_size=1, validation_split=0.3)

#load weights
# model.load_weights("../_data/_save/save_test/weight_test01.h5")

#load model with weight
# model = load_model("../_data/_save/save_test/model_with_weight01.h5")

#ModelCheckpoint load
model = load_model('../_data/_save/save_test/mcp/mcp_test_76-0.079240.h5')


# 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = np.round(model.predict(x_test))

acc = accuracy_score(y_test, y_predict)
print("loss : ", loss[0])
print("acc :",loss[1])
