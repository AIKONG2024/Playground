# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from obesity04_utils import save_submit, save_model, save_csv
from obesity00_constant import SEED, ITERATTIONS, PATIENCE, N_TRIAL, N_SPLIT
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#====================================================================================
#obtuna Tunner 이용
def obtuna_tune():
    import optuna
    
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")
    
    train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height'] ** 2)
    test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height'] ** 2)
    
    lbe= LabelEncoder()
    train_gender = lbe.fit_transform(train_csv['Gender'])
    test_gender = lbe.transform(test_csv['Gender'])
    train_csv['BFP'] = (1.2*train_csv['BMI']) + (0.23*train_csv['Age']) - ((train_gender * 1) + (train_gender + 1)) * 5.4
    test_csv['BFP'] = (1.2*test_csv['BMI']) + (0.23*test_csv['Age']) - ((test_gender * 1) + (test_gender + 1)) * 5.4
    
    train_gender = lbe.fit_transform(train_csv['Gender'])
    test_gender = lbe.transform(test_csv['Gender'])
    
    features = list(train_csv.columns)
    to_remove = ['id','SMOKE']
    [features.remove(feature) for feature in to_remove if feature in to_remove]
    
    # cat_features = train_csv.select_dtypes(include='object').columns.values
    # for feature in cat_features :
    #     train_csv[feature] = train_csv[feature].astype('category')
    #     test_csv[feature] = test_csv[feature].astype('category')
    cat_features = train_csv.select_dtypes(include='object').columns.values
    for feature in cat_features :
        train_csv[feature] = lbe.fit_transform(train_csv[feature])
        if "Always" not in lbe.classes_:
            lbe.classes_ = np.append(lbe.classes_, "Always")
        if feature != "NObeyesdad" :
            test_csv[feature] = lbe.transform(test_csv[feature])
                
    X, y = train_csv.drop(["NObeyesdad"], axis=1), train_csv.NObeyesdad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    
    from sklearn.preprocessing import MaxAbsScaler
    X_train = MaxAbsScaler().fit_transform(X_train)
    X_test = MaxAbsScaler().fit_transform(X_test)
    test_csv = MaxAbsScaler().fit_transform(test_csv)
    
    # Hyperparameter Optimization
    # https://velog.io/@highway92/XGBoost-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%EB%93%A4
    # https://www.kaggle.com/code/abdelrhmanelhelaly/91-5-accuracy
    # https://www.kaggle.com/code/gabedossantos/eda-xgboost-91-5
    def objective(trial: optuna.Trial):
        params = {
            'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'gamma' : trial.suggest_float('gamma', 1e-9, 0.5),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'max_depth': trial.suggest_int('max_depth', 0, 16),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True), 
            'objective' : trial.suggest_categorical('objective', ['multi:sotfmax', 'multi:softprob']) ,
            'eval_metric' :  trial.suggest_categorical('eval_metric', ["merror",'mlogloss', 'auc']),
            'booster' : 'gbtree',
            'verbosity' : 0,
            'device' : 'cuda',
            'tree_method' : 'hist',
            'enable_categorical' : True,
            # 'max_cat_to_onehot' : 1,
            'early_stopping_rounds' : patience,
            # 'importance_type' : 'weight',
            'random_state' : SEED,
        }
        
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],verbose=False)
                
        predictions = clf.predict(X_test)
        return accuracy_score(y_test, predictions)

    study = optuna.create_study(study_name="obesity-accuracy", direction="maximize")
    study.optimize(objective, n_trials=n_trial)
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
    best_model = XGBClassifier(**best_study.params)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],verbose=False)
    predictions = best_model.predict(test_csv)
    submission_csv = pd.read_csv(path + "sample_submission.csv")
    submission_csv["NObeyesdad"] = lbe.inverse_transform(predictions) 
    
    save_csv(path, f"{round(best_study.value,4)}_xgb_", submission_csv)
    save_model(path, f"{round(best_study.value,4)}_xgb_", best_model)

patience = PATIENCE
iterations = ITERATTIONS
n_trial = N_TRIAL
n_splits = N_SPLIT

#====================================================================================

# RUN
def main():
    obtuna_tune()

if __name__ == '__main__':
    main()
    
    # Trial 12 finished with value: 0.9150610147719974 and parameters: {'grow_policy': 'lossguide', 'n_estimators': 996, 'learning_rate': 0.022115905415387556, 'gamma': 0.1299998264675535, 'subsample': 0.7591158724763973, 
    # 'colsample_bytree': 0.3015166580857436, 'max_depth': 7, 'min_child_weight': 7, 'reg_lambda': 0.035320295955169626, 'reg_alpha': 1.645889988069947, 'objective': 'multi:softprob', 'eval_metric': 'auc'}. Best is trial 12 with value: 0.9150610147719974.