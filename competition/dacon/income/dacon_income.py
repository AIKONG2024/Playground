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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

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

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-9, 0.5),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'max_depth': trial.suggest_int('max_depth', 0, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),
        'verbosity': 0,
        'random_state': SEED,
        'tree_method': 'hist',
        'enable_categorical': True,
    }
    
    skf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
    
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, eval_metric="rmse", verbose=False)
        predictions = model.predict(X_val)
        cv_scores[idx] = np.sqrt(mean_squared_error(y_val, predictions))  # RMSE 계산
    
    return np.mean(cv_scores)

study = optuna.create_study(study_name="obesity-accuracy", direction="minimize")
study.optimize(objective, n_trials=50)
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

# predict
best_param = best_study.params
model = XGBRegressor(**best_param)
model.set_params(enable_categorical = True)
model.fit(X, y, eval_set=[(X_test, y_test)], verbose=False)
print(model.score(X_test, y_test))
model.predict(X_test)


submission_csv = pd.read_csv(path + "sample_submission.csv")
predictions = model.predict(test_csv)
submission_csv["Income"] = predictions
submission_csv.to_csv(path + "sample_submission_pred2.csv", index=False)
