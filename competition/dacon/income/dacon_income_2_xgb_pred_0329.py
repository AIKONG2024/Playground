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


#추가 컬럼 생성

X = train_df.drop(['Income'], axis=1)
y = train_df['Income']

#데이터 증폭


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


best_param = {'n_estimators': 732, 'learning_rate': 0.03492987728008141, 'gamma': 0.17226478172732684, 'subsample': 0.8608958630474867, 'colsample_bytree': 0.48326896246422896, 'max_depth': 10, 'min_child_weight': 4, 'reg_lambda': 23.081327860519973, 'reg_alpha': 2.1800615374184198e-05, 'random_state': 19355834}
model = XGBRegressor(**best_param)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, eval_metric="rmse", verbose=False)
predictions = model.predict(X_test)
predictions = np.where(predictions < 0, -predictions, predictions)
score = np.sqrt(mean_squared_error(y_test, predictions))  # RMSE 계산
print('[score] : ', score)

import time
timestr = time.strftime("%Y%m%d%H%M%S")
save_name = timestr

submission_csv = pd.read_csv(path + "sample_submission.csv")
predictions = model.predict(test_df)
predictions = np.where(predictions < 0, -predictions, predictions)
submission_csv["Income"] = predictions
submission_csv.to_csv(path + f"sample_submission_pred{save_name}_{score}.csv", index=False)
