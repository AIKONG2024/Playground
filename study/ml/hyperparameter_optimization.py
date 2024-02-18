import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# #모델구성
# model = SVC(kernel='rbf', C=10, gamma=0.00001, degree=10, max_iter=10) #parameters = kernel , C, gamma, degree, max_iter

# #훈련
# model.fit(X_train, y_train)

# #평가 예측

# predict = np.round(model.predict(X_test))
# acc_score = accuracy_score(y_test, predict)
# print(f'''
#     {type(model).__name__} predict is [{predict}]
#     accuracy score is [{ acc_score}]"
#     ''')

# '''
#     SVC predict is [[0 1 0 1 1 0 0 1 1 1 1 1 0 1 0 1 1 2 1 0 1 1 0 0 1 1 0 1 1 1 0 0 1 1 1 1]]
#     accuracy score is [0.6944444444444444]"
# '''

# '''
#     SVC predict is [[0 2 1 1 2 0 0 1 2 2 1 2 0 2 0 1 1 2 2 0 1 2 0 0 2 1 0 2 2 2 0 1 1 2 2 2]]
#     accuracy score is [0.75]"
# '''


########################################
# GridSearch
# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_wine
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score

# #데이터
# X, y = load_wine(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# #모델구성
# model = SVC() #parameters = kernel , C, gamma, degree, max_iter

# parameters = {'kernel' : ['linear','rbf','poly','sigmoid'],
#               'C' : [1,3,5,10],
#               'degree' : [5, 10, 15, 20],
#               'max_iter' : [1, 5, 10 ,15, 20]}
# gscv = GridSearchCV(estimator=model, param_grid= parameters, cv= 5, n_jobs= -1)

# #훈련
# gscv.fit(X_train, y_train)

# #평가 예측
# best_model = gscv.best_estimator_
# best_param = gscv.best_params_
# predict = np.round(best_model.predict(X_test))
# acc_score = accuracy_score(y_test, predict)
# print(f'''
#     best parameters : {best_param}
#     {type(model).__name__} predict is [{predict}]
#     accuracy score is [{ acc_score}]"
#     ''')
# '''
#     best parameters : {'C': 10, 'degree': 5, 'kernel': 'rbf', 'max_iter': 15}
#     SVC predict is [[0 2 0 1 2 0 0 1 2 2 1 2 0 2 0 1 1 2 2 0 1 2 0 0 2 1 0 2 2 2 0 0 1 2 2 2]]
#     accuracy score is [0.7777777777777778]"
# '''

########################################
# # RandomSearch
# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_wine
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score

# #데이터
# X, y = load_wine(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# #모델구성
# model = SVC() #parameters = kernel , C, gamma, degree, max_iter

# parameters = {'kernel' : ['linear','rbf','poly','sigmoid'],
#               'C' : [1,3,5,10],
#               'degree' : [5, 10, 15, 20],
#               'max_iter' : [1, 5, 10 ,15, 20]}
# gscv = RandomizedSearchCV(estimator=model, param_distributions= parameters, cv= 5, n_jobs= -1)

# #훈련
# gscv.fit(X_train, y_train)

# #평가 예측
# best_model = gscv.best_estimator_
# best_param = gscv.best_params_
# predict = np.round(best_model.predict(X_test))
# acc_score = accuracy_score(y_test, predict)
# print(f'''
#     best parameters : {best_param}
#     {type(model).__name__} predict is [{predict}]
#     accuracy score is [{ acc_score}]"
#     ''')
# '''
#     best parameters : {'max_iter': 1, 'kernel': 'linear', 'degree': 15, 'C': 5}
#     SVC predict is [[0 2 2 1 2 0 0 1 2 2 1 1 0 2 0 1 1 2 2 0 1 2 0 0 2 1 0 1 2 2 0 2 1 1 2 2]]
#     accuracy score is [0.6944444444444444]"
# '''

########################################
# Hyperopt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# hp.choice : 리스트 내의 값을 무작위로 추출
# hp.uniform : 정의된 범위 내의 임의 숫자 추출
# hp.quniform : 정의된 범위에서 간격을 참고하여 숫자 추출
space = {
    'C': hp.uniform('C', 0.1, 10.0),
    "degree": hp.quniform("degree", 1, 10, 1),
    "max_iter": hp.quniform("max_iter", 1, 100, 10),
}
def objective(space):
    model = SVC(
               C = space['C'],
               degree= space['degree'],
               max_iter= space['max_iter']
               )
    return {'loss': -1, 'status' : STATUS_OK }

trials = Trials()
best_param = fmin(fn=objective,
                  space=space,
                  algo=tpe.suggest,
                  max_evals=10,
                  trials= trials)
# 재훈련
print(best_param)
model = SVC(**best_param)
model.fit(X_train, y_train)

# #평가 예측
predict = np.round(model.predict(X_test))
acc_score = accuracy_score(y_test, predict)
print(f'''
    best parameters : {best_param}
    {type(model).__name__} predict is [{predict}]
    accuracy score is [{ acc_score}]"
    ''')

'''
    best parameters : {'C': 8.863929212636368, 'degree': 5, 'max_iter': 21}
    SVC predict is [[0 2 0 1 2 0 0 1 2 2 1 2 0 2 0 1 1 2 2 0 1 2 0 0 2 1 0 2 2 2 0 0 1 2 2 2]]
    accuracy score is [0.7777777777777778]"
'''

########################################
# Optuna
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import optuna

def objective(trial: optuna.Trial):
    params = {
    'C': trial.suggest_uniform('C', 0.1, 10.0),
    "degree":  trial.suggest_int("degree", 1, 10, 1),
    "max_iter":  trial.suggest_int("max_iter", 1, 100, 10),
    }

    svc = SVC(**params)
    svc.fit(X_train, y_train)
    predictions = svc.predict(X_test)
    return accuracy_score(y_test, predictions)

study = optuna.create_study(study_name="obesity-accuracy", direction="maximize")
study.optimize(objective, n_trials=10)
best_study = study.best_trial
best_param = best_study.params

# 재훈련
model = SVC(**best_param)
model.fit(X_train, y_train)
# #평가 예측
predict = np.round(model.predict(X_test))
acc_score = accuracy_score(y_test, predict)
print(f'''
    best parameters : {best_param}
    {type(model).__name__} predict is [{predict}]
    accuracy score is [{ acc_score}]"
    ''')
'''
    best parameters : {'C': 7.544583675807302, 'degree': 5, 'max_iter': 21}
    SVC predict is [[0 2 0 1 2 0 0 1 2 2 1 2 0 2 0 1 1 2 2 0 1 2 0 0 2 1 0 2 2 2 0 0 1 2 2 2]]
    accuracy score is [0.7777777777777778]"
'''