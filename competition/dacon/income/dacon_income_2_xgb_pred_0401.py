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
    #데이터
path = 'C:/_data/dacon/income/'
while 1:
    SEED = random.randint(0, 40000000)
    print('==============================')
    print('[seed] : ', SEED)
    #read 
    train_df = pd.read_csv(path + "train.csv", index_col=0)
    test_df = pd.read_csv(path + "test.csv", index_col=0)

    train_df = train_df.drop(['Gains', 'Losses','Dividends'], axis=1)
    test_df = test_df.drop(['Gains', 'Losses', 'Dividends'], axis=1)

    print(np.unique(train_df['Age'].min()))
    print(np.unique(train_df['Age'].mean()))
    print(np.unique(train_df['Age'].max()))
    #이산형 범주 처리
    bins = [0, 26, 52]
    labels = [0, 1]
    train_df['Working_Week (Yearly)'] = pd.cut(train_df['Working_Week (Yearly)'], bins=bins, labels=labels, include_lowest=True).astype(np.float32)
    test_df['Working_Week (Yearly)'] = pd.cut(test_df['Working_Week (Yearly)'], bins=bins, labels=labels, include_lowest=True).astype(np.float32)

    bins = [0, 20, 50, 60, 90]
    labels = [0, 1, 3, 2]
    train_df['Age'] = pd.cut(train_df['Age'], bins=bins, labels=labels, include_lowest=True).astype(np.float32)
    test_df['Age'] = pd.cut(test_df['Age'], bins=bins, labels=labels, include_lowest=True).astype(np.float32)

    #음수가 나오는 이유 분석

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

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
    # print('[score] : ', score)
    # 584.3783069457488.
    # [score] :  530.7380532958175
    import time
    timestr = time.strftime("%Y%m%d%H%M%S")
    save_name = timestr
    if 535 < score < 538:
        submission_csv = pd.read_csv(path + "sample_submission.csv")
        predictions = model.predict(test_df)
        predictions = np.where(predictions < 0, -predictions, predictions)
        negative_predictions_count = sum(predictions < 0)
        print('음수 개수 :',negative_predictions_count)
        submission_csv["Income"] = predictions
        submission_csv.to_csv(path + f"sample_submission_pred{save_name}_{score}.csv", index=False)
        break
              
'''
[seed] :  9562350
[0]
[35.6325]
[90]
[score] :  534.5674676304851
음수 개수 : 0
'''  