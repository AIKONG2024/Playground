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
def outliers(data):
    if np.issubdtype(data.dtype, np.number):  # 수치형 데이터인지 확인
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return np.where((data < lower_bound) | (data > upper_bound))
    else:
        return np.array([])

import pandas as pd
import numpy as np
from  xgboost import XGBRegressor
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

path = 'C:/_data/dacon/income/'
SEED = 42
N_SPLITS = 8
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

print(train_csv.shape) #(20000, 22)
print(test_csv.shape) #(10000, 21)

# print(train_csv.Income_Status.head(50))
lbe = LabelEncoder()

print(np.unique(train_csv['Household_Status']))
print("===================")
print(np.unique(test_csv['Household_Status'].astype(str)))
print("===================")

categorical_features = train_csv.select_dtypes(include='object').columns.values
lbe = LabelEncoder()
for feature in categorical_features:
    lbe.fit(train_csv[feature])
    train_csv[feature] = lbe.transform(train_csv[feature])    
    for label in test_csv[feature]: 
        if label not in lbe.classes_:
            lbe.classes_ = np.append(lbe.classes_,label)
            print(lbe.classes_)   
    test_csv[feature] = lbe.transform(test_csv[feature])

X = train_csv.drop(["Income"], axis=1)
y = train_csv['Income']

#이상치 제거
outlier_indices = set()
for col in X.columns:
    if X[col].dtype == 'float64' or X[col].dtype == 'int64':
        outliers_idx = outliers(X[col])[0]
        outlier_indices.update(outliers_idx)
outlier_indices_list = list(outlier_indices) 
X_filtered = X.drop(index=outlier_indices_list, errors='ignore')  
y_filtered = y.loc[X_filtered.index] 

# 결측치 제거
X = X.interpolate(method='linear', limit_direction='forward', axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
####################################
# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
#         'gamma': trial.suggest_float('gamma', 1e-9, 0.5),
#         'subsample': trial.suggest_float('subsample', 0.3, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
#         'max_depth': trial.suggest_int('max_depth', 0, 16),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),
#         'verbosity': 0,
#         'random_state': trial.suggest_int('random_state', 0, 300000),
#         'tree_method': 'hist',
#         'enable_categorical': True,
#     }
    
#     # skf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
#     # cv_scores = np.empty(N_SPLITS)
#     # for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
#     #     X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
#     #     y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
    
#     #     model = XGBRegressor(**params)
#     #     model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, eval_metric="rmse", verbose=False)
#     #     predictions = model.predict(X_val)
#     #     cv_scores[idx] = np.sqrt(mean_squared_error(y_val, predictions))  # RMSE 계산
#     model = XGBRegressor(**params)
#     model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, eval_metric="rmse", verbose=False)
#     predictions = model.predict(X_test)
#     cv_scores = np.sqrt(mean_squared_error(y_test, predictions))  # RMSE 계산
#     return np.mean(cv_scores)

# study = optuna.create_study(study_name="obesity-accuracy", direction="minimize")
# study.optimize(objective, n_trials=300)
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
# model = XGBRegressor(**best_param)
# model.set_params(enable_categorical = True)
# model.fit(X, y, eval_set=[(X_test, y_test)], verbose=False)
# print(model.score(X_test, y_test))
# model.predict(X_test)
###########################################################
# model = RandomForestRegressor()
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
# pred = model.predict(X_test)
# print(f'rmse : {np.sqrt(mean_squared_error(pred, y_test))}')
#######################################
model = LGBMRegressor(random_state=SEED)
model.fit(X, y, eval_set=[(X_test, y_test)])
print(model.score(X_test, y_test))
pred = model.predict(X_test)
score = np.sqrt(mean_squared_error(pred, y_test))
print(f'rmse : {score}')

import time
timestr = time.strftime("%Y%m%d%H%M%S")
save_name = timestr

submission_csv = pd.read_csv(path + "sample_submission.csv")
predictions = model.predict(test_csv)
submission_csv["Income"] = predictions
submission_csv.to_csv(path + f"sample_submission_pred{save_name}_{score}.csv", index=False)
