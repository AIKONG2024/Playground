#https://dacon.io/competitions/official/236230/overview/description
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna

N_SPLITS = 5

def outliers(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return (data > upper_bound) | (data < lower_bound)

#데이터
path = 'C:/_data/dacon/income/'
SEED = 42

#read 
train_df = pd.read_csv(path + "train.csv", index_col=0)
test_df = pd.read_csv(path + "test.csv", index_col=0)

train_df = train_df.drop(['Gains', 'Losses','Dividends'], axis=1)
test_df = test_df.drop(['Gains', 'Losses', 'Dividends'], axis=1)

#그 외 범주형 데이터
lbe = LabelEncoder()
categorical_features = [col for col in train_df.columns if train_df[col].dtype == 'object']
for feature in categorical_features:
    lbe.fit(train_df[feature])
    train_df[feature] = lbe.transform(train_df[feature])
    new_labels = set(test_df[feature]) - set(lbe.classes_)
    for new_label in new_labels:
        lbe.classes_ = np.append(lbe.classes_, new_label)
    test_df[feature] = lbe.transform(test_df[feature])

#데이터 증폭

#추가 컬럼 생성

X = train_df.drop(['Income'], axis=1)
y = train_df['Income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

#스케일링
#################################################################
####################################

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-9, 0.5),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'max_depth': trial.suggest_int('max_depth', 0, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),
        'verbosity': 0,
        'random_state': trial.suggest_int('random_state', 0, 30000000),
        'tree_method': 'hist',
        'enable_categorical': True,
    }
    
    skf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cv_scores = np.empty(N_SPLITS)
    for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
    
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        predictions = model.predict(X_val)
        cv_scores[idx] = np.sqrt(mean_squared_error(y_val, predictions))  # RMSE 계산
    # model = LGBMRegressor(**params)
    # model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="rmse")
    # predictions = model.predict(X_test)
    # cv_scores = np.sqrt(mean_squared_error(y_test, predictions))  # RMSE 계산
    return np.mean(cv_scores)

study = optuna.create_study(study_name="obesity-accuracy", direction="minimize")
study.optimize(objective, n_trials=200)
best_study = study.best_trial

# print(
# f"""
# ============================================
# [Trials completed : {len(study.trials)}]
# [Best params : {best_study.params}]
# [Best value: {best_study.value}]
# ============================================
# """
# )
param = best_study.params
# param = {'learning_rate': 0.11996095970505455, 'subsample': 0.8885386193013425, 'min_child_weight': 0.33695048709449, 'num_leaves': 13}
model = LGBMRegressor(**param, random_state=SEED)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
pred = model.predict(X_test)
score = np.sqrt(mean_squared_error(pred, y_test))
print(f'rmse : {score}')
import time
timestr = time.strftime("%Y%m%d%H%M%S")
save_name = timestr

submission_csv = pd.read_csv(path + "sample_submission.csv")
predictions = model.predict(test_df)
submission_csv["Income"] = predictions
submission_csv.to_csv(path + f"sample_submission_pred{save_name}_{score}.csv", index=False)

# model = LGBMRegressor(random_state=SEED)
# model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
# pred = model.predict(X_test)
# score = np.sqrt(mean_squared_error(pred, y_test))
# print(f'rmse : {score}')
# rmse : 590.1690312701155

'''
[Best params : {'learning_rate': 0.11996095970505455, 'subsample': 0.8885386193013425, 'min_child_weight': 0.33695048709449, 'num_leaves': 13}]
[Best value: 588.9691820202924]
'''