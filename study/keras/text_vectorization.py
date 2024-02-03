import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Embedding


X = np.array(["cat","mat", "on"])

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
x_sequences = np.array(tokenizer.texts_to_sequences(X)) 
print(x_sequences.shape) #(3, 1)

embedding_layer = Embedding(input_dim=9, output_dim=4)
result = np.array(embedding_layer(x_sequences)) 
print(result.shape)#(3, 1, 4)
print(result)

'''
[[[-0.0170779   0.03554878  0.03025447 -0.00052363]]

 [[ 0.02581792 -0.02460885 -0.03337549  0.00473196]]

 [[-0.04555179 -0.01618879 -0.02047335 -0.03256543]]]
'''



#========================임베딩
# X = np.array(["달","밝은", "밤이면", "창가에", "흐르는", "내", "젊은","연가가", "구슬퍼"])
# y = np.array([1,2,3,4,5,6,7,8,9])

# from keras.preprocessing.text import Tokenizer

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X)
# x_sequences = np.array(tokenizer.texts_to_sequences(X)) 
# print(x_sequences.shape) #(9, 1)

# embedding_layer = Embedding(input_dim=9, output_dim=5)
# result = np.array(embedding_layer(x_sequences)) 
# print(result.shape)#(9, 1, 5)
# print(result)

# '''
# [[[ 0.00477131 -0.04287178  0.00361929  0.01333213 -0.02192663]]

#  [[-0.00837155  0.00626709  0.04374525 -0.03213976  0.0266226 ]]

#  [[ 0.00735215  0.02038765 -0.0424327  -0.0192417  -0.02178895]]

#  [[ 0.03623239  0.01762425 -0.00429205  0.01020346  0.02606006]]

#  [[-0.0498311  -0.03270175 -0.00377637  0.03170151 -0.04811602]]

#  [[-0.03430237  0.03651917 -0.00129032  0.01626677  0.00634655]]

#  [[-0.00240629 -0.01048706  0.0080834   0.01493727  0.01620818]]

#  [[ 0.02655579  0.03729847 -0.03375788  0.00081546  0.01415124]]

#  [[ 0.          0.          0.          0.          0.        ]]]
# '''


# ===========================원핫인코딩
# ohe = OneHotEncoder(sparse=False)
# x_train = ohe.fit_transform(x_train.reshape(-1,1))
# x_test = ohe.fit_transform(x_test.reshape(-1,1))
# print(X)
# '''
# [[0. 0. 1. 0. 0. 0. 0. 0. 0.]   "달"
#  [0. 0. 0. 1. 0. 0. 0. 0. 0.]   "밝은"    
#  [0. 0. 0. 0. 1. 0. 0. 0. 0.]   "밤이면"
#  [0. 0. 0. 0. 0. 0. 0. 1. 0.]   "창가에"
#  [0. 0. 0. 0. 0. 0. 0. 0. 1.]   "흐르는"
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]   "내"
#  [0. 0. 0. 0. 0. 0. 1. 0. 0.]   "젊은"
#  [0. 0. 0. 0. 0. 1. 0. 0. 0.]   "연가가"
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]]  "구슬퍼"
# '''
# print(x_test.shape) #(9, 9)
# x_train = np.expand_dims(X, axis=2) #RNN에 넣어주기 위해 3차원 변경 (9, 9, 1)
# print(X.shape) #(9, 9, 1)

# from keras.models import Sequential
# from keras.layers import LSTM, Dense

# model = Sequential()

# model.add(
#     LSTM(
#         units=10,
#         activation="tanh",
#         recurrent_activation="sigmoid",
#         dropout=0,
#         recurrent_dropout=0,
#         unroll=False,
#         use_bias=True,
#         input_shape=(9,1)
#     )
# )
# model.add(Dense(32,activation='relu'))
# model.add(Dense(1, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam')
# model.fit(x_train, y_train, epochs=200)

# x_predict = ohe.inverse_transform(model.predict(x_test)) 
# print(x_predict)