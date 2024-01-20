import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

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

# 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=30))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1, activation="sigmoid"))

# # save model to json
# json_string = model.to_json()
# # model.to_yaml()
# with open('../_data/_save/save_test/json_model01.json', 'w') as f:
#     f.write(json_string)

# # save model only
# model.save("../_data/_save/save_test/model01.h5")

#ModelCheckpoint save
mch = ModelCheckpoint(
    filepath="../_data/_save/save_test/mcp/mcp_test_{epoch:02d}-{val_loss:2f}.h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    initial_value_threshold=0.083 #저장의 기준 수치
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
history = model.fit(
    x_train, y_train, epochs=76, batch_size=1000, validation_split=0.3, callbacks=[mch]
)

# #save weight
# model.save_weights("../_data/_save/save_test/weight_test01.h5")


# #save model with weight
# model.save("../_data/_save/save_test/model_with_weight01.h5")


# 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = np.round(model.predict(x_test))

acc = accuracy_score(y_test, y_predict)
print(loss[0])
print(loss[1])
