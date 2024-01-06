# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# 모델을 만드는 함수
def create_model():
    model = Sequential()
    model.add(Dense(32, input_dim=10, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

# KerasRegressor로 모델 감싸기
model = KerasRegressor(build_fn=create_model, verbose=0)

# GridSearchCV를 위한 매개변수 정의
param_grid = {'batch_size': [10, 20, 40, 60, 80, 100],
              'epochs': [10, 50, 100]}

# GridSearchCV 생성
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_train, y_train)

# 결과 출력
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))