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
import random

N_SPLITS = 5

#데이터
path = 'C:/_data/dacon/income/'
SEED = 42
print('==============================')
print('[seed] : ', SEED)
#read 
train_df = pd.read_csv(path + "train.csv", index_col=0)
test_df = pd.read_csv(path + "test.csv", index_col=0)

train_df = train_df.drop(['Gains', 'Losses','Dividends'], axis=1)
test_df = test_df.drop(['Gains', 'Losses', 'Dividends'], axis=1)

#이산형 범주 처리
bins = [0, 26, 52]
labels = [0, 1]
train_df['Working_Week (Yearly)'] = pd.cut(train_df['Working_Week (Yearly)'], bins=bins, labels=labels, include_lowest=True).astype(np.float32)
test_df['Working_Week (Yearly)'] = pd.cut(test_df['Working_Week (Yearly)'], bins=bins, labels=labels, include_lowest=True).astype(np.float32)

#인코딩
#범주형 데이터
lbe = LabelEncoder()
categorical_features = [col for col in train_df.columns if train_df[col].dtype == 'object']
for feature in categorical_features:
    lbe.fit(train_df[feature])
    train_df[feature] = lbe.transform(train_df[feature])
    new_labels = set(test_df[feature]) - set(lbe.classes_)
    for new_label in new_labels:
        lbe.classes_ = np.append(lbe.classes_, new_label)
    test_df[feature] = lbe.transform(test_df[feature])

X = train_df.drop(['Income'], axis=1)
y = train_df['Income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

best_param = {'n_estimators': 732, 'learning_rate': 0.01492987728008141, 'gamma': 0.17226478172732684, 'subsample': 0.8608958630474867, 'colsample_bytree': 0.48326896246422896, 'max_depth': 10, 'min_child_weight': 4, 'reg_lambda': 23.081327860519973, 'reg_alpha': 2.1800615374184198e-05, 'random_state': 19355834}
model = XGBRegressor(**best_param)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, eval_metric="rmse", verbose=False)
predictions = model.predict(X_test)
# predictions = np.exp(predictions) - 1 #로그 역변환
score = np.sqrt(mean_squared_error(y_test, predictions))  # RMSE 계산
print('[score] : ', score)

import time
timestr = time.strftime("%Y%m%d%H%M%S")
save_name = timestr
if score < 540:
    submission_csv = pd.read_csv(path + "sample_submission.csv")
    predictions = model.predict(test_df)
    submission_csv["Income"] = predictions
    submission_csv.to_csv(path + f"sample_submission_pred{save_name}_{score}.csv", index=False)



#best 583.59 
# best_param = {'n_estimators': 732, 'learning_rate': 0.03492987728008141, 'gamma': 0.17226478172732684, 'subsample': 0.8608958630474867, 'colsample_bytree': 0.48326896246422896, 'max_depth': 10, 'min_child_weight': 4, 'reg_lambda': 23.081327860519973, 'reg_alpha': 2.1800615374184198e-05, 'random_state': 19355834}


'''
[Trials completed : 300]
[Best params : {'n_estimators': 315, 'learning_rate': 0.023446565815844873, 'gamma': 0.11389851525261992, 'subsample': 0.754137045493653, 'colsample_bytree': 0.7175982112732845, 'max_depth': 13, 'min_child_weight': 1, 'reg_lambda': 55.02660107515273, 'reg_alpha': 0.0006824903229650601, 'random_state': 40304}]
[Best value: 584.4412672287493]
'''

'''
[Trials completed : 200]
[Best params : {'n_estimators': 364, 'learning_rate': 0.042138693131428484, 'gamma': 0.28934299932110186, 'subsample': 0.5416897123220303, 'colsample_bytree': 0.7409480998899669, 'max_depth': 13, 'min_child_weight': 3, 'reg_lambda': 12.409399889967816, 'reg_alpha': 0.2476467950346773, 'random_state': 26646641}]
[Best value: 584.5606183302049]
'''

'''
[Trials completed : 200]
[Best params : {'n_estimators': 732, 'learning_rate': 0.03492987728008141, 'gamma': 0.17226478172732684, 'subsample': 0.8608958630474867, 'colsample_bytree': 0.48326896246422896, 'max_depth': 10, 'min_child_weight': 4, 'reg_lambda': 23.081327860519973, 'reg_alpha': 2.1800615374184198e-05, 'random_state': 19355834}]
[Best value: 583.6651997791523]
'''