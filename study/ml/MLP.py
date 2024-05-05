import numpy as np
# from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])
for _ in range(0,20):
    model = Perceptron()

    model.fit(x, y)
    predict = model.predict(x)
    acc_score = accuracy_score(y, predict)
    print(f'''
        {type(model).__name__} predict is [{predict}]
        accuracy score is [{ acc_score}]"
        ''')
    
########################################################
 
import numpy as np   
from keras.models import Model
from keras.layers import Input,Dense
from sklearn.metrics import accuracy_score
    
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

input_layer = Input(shape= (2,))
hidden_layer = Dense(10, activation='relu')(input_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)

model = Model(inputs = input_layer, outputs = output_layer)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=100, verbose=0)

predict = np.round(model.predict(x, verbose=0)).reshape(-1,).astype(int)
acc_score = accuracy_score(y, predict)
print(f'''
    {type(model).__name__} predict is [{predict}]
    accuracy score is [{ acc_score}]"
    ''')
##########################################################

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])
for _ in range(0,20):
    model = SVC(kernel='rbf', C=1)

    model.fit(x, y)
    predict = model.predict(x)
    acc_score = accuracy_score(y, predict)
    print(f'''
        {type(model).__name__} predict is [{predict}]
        accuracy score is [{ acc_score}]"
        ''')

