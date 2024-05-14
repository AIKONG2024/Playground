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

#!!!
#Martial_Status - 순서형
# print(np.unique(train_df['Citizenship']))

#인코딩
# 명목형 데이터 처리
nominal_columns = ['Gender', 'Race', 'Employment_Status', 'Industry_Status','Occupation_Status','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)']
one_hot_encoder = OneHotEncoder(sparse=False)
ohe_train = one_hot_encoder.fit_transform(train_df[nominal_columns])
ohe_test = one_hot_encoder.transform(test_df[nominal_columns])
nominal_train_df = pd.DataFrame(ohe_train, columns=one_hot_encoder.get_feature_names_out(nominal_columns), index=train_df.index)
nominal_test_df = pd.DataFrame(ohe_test, columns=one_hot_encoder.get_feature_names_out(nominal_columns), index=test_df.index)

# 순서형 데이터 처리
# ordinal_columns = ['Education_Status', 'Income_Status', 'Martial_Status', 'Citizenship']
# education_order = [
#      'Children', 'Kindergarten', 'Elementary (1-4)', 'Elementary (5-6)', 'Middle (7-8)', 
#     'High Freshman', 'High Sophomore', 'High Junior', 'High Senior', 'High graduate','College', 
#     'Associates degree (Academic)', 'Associates degree (Vocational)',
#     'Bachelors degree', 'Masters degree', 'Professional degree', 'Doctorate degree'
# ]
# income_order = ['Unknown', 'Over Median', 'Under Median', 'Citizenship']
# martial_staus_order = ['Single', 'Separated','Divorced', 'Widowed', 'Married (Armed Force Spouse)', 'Married','Married (Spouse Absent)']
# citizen_order = ['Foreign-born (Naturalized US Citizen)', 'Foreign-born (Non-US Citizen)', 'Native (Born in Puerto Rico or US Outlying)','Native (Born Abroad)', 'Native']
# oe = OrdinalEncoder(categories=[education_order, income_order, martial_staus_order, citizen_order])
# oe_train = oe.fit_transform(train_df[ordinal_columns])
# oe_test = oe.transform(test_df[ordinal_columns])
# ordinal_train_df = pd.DataFrame(oe_train, columns=ordinal_columns, index=train_df.index)
# ordinal_test_df = pd.DataFrame(oe_test, columns=ordinal_columns, index=test_df.index)
# train_df = pd.concat([train_df.drop(columns=nominal_columns+ordinal_columns), nominal_train_df, ordinal_train_df], axis=1)
# test_df = pd.concat([test_df.drop(columns=nominal_columns+ordinal_columns), nominal_test_df, ordinal_test_df], axis=1)

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

#PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=1)
# train_df = pca.fit_transform(train_df)
# test_df = pca.transform(test_df)

#추가 컬럼 생성

X = train_df.drop(['Income'], axis=1)
y = train_df['Income']

#데이터 증폭

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
    
    # skf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    # cv_scores = np.empty(N_SPLITS)
    # for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    #     X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    #     y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
    
    #     model = XGBRegressor(**params)
    #     model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, eval_metric="rmse", verbose=False)
    #     predictions = model.predict(X_val)
    #     cv_scores[idx] = np.sqrt(mean_squared_error(y_val, predictions))  # RMSE 계산
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, eval_metric="rmse", verbose=False)
    predictions = model.predict(X_test)
    cv_scores = np.sqrt(mean_squared_error(y_test, predictions))  # RMSE 계산
    return np.mean(cv_scores)

study = optuna.create_study(study_name="obesity-accuracy", direction="minimize")
study.optimize(objective, n_trials=200)
best_study = study.best_trial

print(
f"""
============================================
[Trials completed : {len(study.trials)}]
[Best params : {best_study.params}]
[Best value: {best_study.value}]
============================================
"""
)
# predict
best_param = best_study.params

model = XGBRegressor(**best_param)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, eval_metric="rmse", verbose=False)
predictions = model.predict(X_test)
score = np.sqrt(mean_squared_error(y_test, predictions))  # RMSE 계산
print('[score] : ', score)

# 584.3783069457488.
# [score] :  530.7380532958175
# import time
# timestr = time.strftime("%Y%m%d%H%M%S")
# save_name = timestr

# submission_csv = pd.read_csv(path + "sample_submission.csv")
# predictions = model.predict(test_df)
# submission_csv["Income"] = predictions
# submission_csv.to_csv(path + f"sample_submission_pred{save_name}_{score}.csv", index=False)

# model = LGBMRegressor(random_state=SEED)
# model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
# pred = model.predict(X_test)
# score = np.sqrt(mean_squared_error(pred, y_test))
# print(f'rmse : {score}')
# rmse : 590.1690312701155
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