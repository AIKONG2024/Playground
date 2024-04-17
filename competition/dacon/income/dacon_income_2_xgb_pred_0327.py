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

def outliers(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return (data > upper_bound) | (data < lower_bound)
while 1 :
    #데이터
    path = 'C:/_data/dacon/income/'
    SEED = random.randint(0,112)
    print('==============================')
    print('[seed] : ', SEED)
    #read 
    train_df = pd.read_csv(path + "train.csv", index_col=0)
    test_df = pd.read_csv(path + "test.csv", index_col=0)

    train_df = train_df.drop(['Gains', 'Losses','Dividends'], axis=1)
    test_df = test_df.drop(['Gains', 'Losses', 'Dividends'], axis=1)

    #이산형 범주 처리
    # bins = [0, 26, 52]
    # labels = [0, 1]
    # train_df['Working_Week (Yearly)'] = pd.cut(train_df['Working_Week (Yearly)'], bins=bins, labels=labels, include_lowest=True).astype(np.float32)
    # test_df['Working_Week (Yearly)'] = pd.cut(test_df['Working_Week (Yearly)'], bins=bins, labels=labels, include_lowest=True).astype(np.float32)

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


    #종속변수 로그
    # train_df['Income'] = np.log(train_df['Income'] + 1)
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

    # study = optuna.create_study(study_name="obesity-accuracy", direction="minimize")
    # study.optimize(objective, n_trials=1000)
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
    best_param = {'n_estimators': 732, 'learning_rate': 0.01492987728008141, 'gamma': 0.17226478172732684, 'subsample': 0.8608958630474867, 'colsample_bytree': 0.48326896246422896, 'max_depth': 10, 'min_child_weight': 4, 'reg_lambda': 23.081327860519973, 'reg_alpha': 2.1800615374184198e-05, 'random_state': 19355834}
    model = XGBRegressor(**best_param)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=90, eval_metric="rmse", verbose=False)
    predictions = model.predict(X_test)
    score = np.sqrt(mean_squared_error(y_test, predictions))  # RMSE 계산
    print('[score] : ', score)
    if 536< score <537:
    # 584.3783069457488.
    # [score] :  530.7380532958175
        import time
        timestr = time.strftime("%Y%m%d%H%M%S")
        save_name = timestr

        submission_csv = pd.read_csv(path + "sample_submission.csv")
        predictions = model.predict(test_df)
        submission_csv["Income"] = predictions
        submission_csv.to_csv(path + f"sample_submission_pred{save_name}_{score}.csv", index=False)
        break
        

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

'''
데이터 전처리 후 
[Trials completed : 400]
[Best params : {'n_estimators': 770, 'learning_rate': 0.015090225620716016, 'gamma': 0.1475815911307591, 'subsample': 0.8643793568717909, 'colsample_bytree': 0.33574074852338315, 'max_depth': 16, 'min_child_weight': 3, 'reg_lambda': 53.325922021619625, 'reg_alpha': 1.8908060018508e-08, 'random_state': 1388287}]
[Best value: 535.8783187384585]

[Trials completed : 400]
[Best params : {'n_estimators': 345, 'learning_rate': 0.03878127099183502, 'gamma': 0.46104505033863724, 'subsample': 0.45960665695955305, 'colsample_bytree': 0.5314120441073775, 'max_depth': 7, 'min_child_weight': 4, 'reg_lambda': 6.5383017715027325, 'reg_alpha': 0.00017043575783917767, 'random_state': 3597472}]
[Best value: 534.7569062903265]

[Trials completed : 400]
[Best params : {'n_estimators': 923, 'learning_rate': 0.05471780822314705, 'gamma': 0.17903461208708793, 'subsample': 0.9081230830471466, 'colsample_bytree': 0.383701862916095, 'max_depth': 7, 'min_child_weight': 4, 'reg_lambda': 11.050171915769123, 'reg_alpha': 0.027005756261642113, 'random_state': 20912043}]
[Best value: 534.6241299412229]
'''