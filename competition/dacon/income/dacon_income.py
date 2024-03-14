#https://dacon.io/competitions/official/236230/overview/description
#문제 분석
#답안 : 수입(Income)
#Age : -
#Gender : 단계별 수치x
#Education_Status : 단계별 수치 
#Employment_Status : 단계별 수치
#Working_Week : unKnown 처리
#Race : 단계별 수치x
#Hispanic_Origin : 단계별 수치x
#Martial_Status : 단계별 수치 x
#Household_Status : 단계별 수치 x
#Household_Summary : 단계별 수치 x
#Citizenship: 단계별 수치 x
#Birth_Country : 단계별 수치 x
#Birth_Country (Father)
#Birth_Country (Mother)
#Tax_Status : 단계별 수치 x
#Gains :
#Losses:
#Divdends:
#Incom_Status: 단계별 수치 
#
import pandas as pd
import numpy as np
from  xgboost import XGBRegressor
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

path = 'C:/_data/dacon/income/'
SEED = 42
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

print(train_csv.shape) #(20000, 22)
print(test_csv.shape) #(10000, 21)

# print(train_csv.Income_Status.head(50))

categorical_features = train_csv.select_dtypes(include='object').columns.values
for feature in categorical_features :
    train_csv[feature] = train_csv[feature].astype('category')
    test_csv[feature] = test_csv[feature].astype('category')

X = train_csv.drop(["Income"], axis=1)
y = train_csv['Income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

#모델 구성
# model = XGBRegressor()

# def objective(trial: optuna.Trial):
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
#             'gamma' : trial.suggest_float('gamma', 1e-9, 0.5),
#             'subsample': trial.suggest_float('subsample', 0.3, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
#             'max_depth': trial.suggest_int('max_depth', 0, 16),
#             'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
#             'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
#             'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True), 
#             'eval_metric' :  'rmse',
#             'verbosity' : 0,
#             'device' : 'cuda',
#             'tree_method' : 'hist',
#             # 'enable_categorical' : True,
#             # 'max_cat_to_onehot' : 1,
#             'early_stopping_rounds' : 10,
#             # 'importance_type' : 'weight',
#             'random_state' : SEED,
#         }
        
#         model = XGBRegressor(**params)
#         model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose= False)
#         predictions = model.predict(X_test)
#         return mean_squared_error(y_test, predictions)

# study = optuna.create_study(study_name="obesity-accuracy", direction="maximize")
# study.optimize(objective, n_trials=100)
# best_study = study.best_trial

# print(
# f"""
# ============================================
# [Trials completed : {len(study.trials)}]
# [Best params : {best_study.params}]
# [Best value: {best_study.value}]
# ============================================
# """
# )

# predict
# best_param = best_study.params
model = XGBRegressor()
model.set_params(enable_categorical = True)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(model.score(X_test, y_test))

submission_csv = pd.read_csv(path + "sample_submission.csv")
predictions = model.predict(test_csv)
submission_csv["Income"] = predictions
submission_csv.to_csv(path + "sample_submission_pred.csv", index=False)
