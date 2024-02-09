# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
from sklearn.metrics import accuracy_score
from obesity01_data import lable_encoding, get_data
from obesity02_models import get_xgboost, get_fitted_xgboost
from obesity04_utils import save

#====================================================================================
#obtuna Tunner 이용
def obtuna_tune():
    import optuna
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")
    features = ['Gender','family_history_with_overweight','FAVC','CAEC',
                                           'SMOKE','SCC','CALC','MTRANS']
    train_csv, test_csv, encoder = lable_encoding(train_csv, test_csv)
    datasets = get_data(train_csv)

    # Hyperparameter Optimization
    # https://velog.io/@highway92/XGBoost-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%EB%93%A4
    def objective(trial: optuna.Trial):
        params = {
            'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'gamma' : trial.suggest_float('gamma', 1e-9, 1.0), #필수
            'subsample': trial.suggest_float('subsample', 0.25, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.25, 1.0),
            'max_depth': trial.suggest_int('max_depth', 0, 24),#필수
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30), #필수 
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),    
            'eval_metric' : 'auc'
        }
        params['booster'] = 'gbtree'
        params["verbosity"] = 0
        clf = get_fitted_xgboost(params, datasets, PATIENCE)
        
        X_test, y_test = datasets[1], datasets[3]
        predictions = clf.predict(X_test)
        return accuracy_score(y_test, predictions)

    study = optuna.create_study(study_name="obesity-accuracy", direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
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
    best_model = get_fitted_xgboost(best_study.params, datasets, PATIENCE)  # bestest
    predictions = encoder.inverse_transform(best_model.predict(test_csv))
    save(path, predictions)

    
#====================================================================================
# GridSearchCV Tunner 이용
def GridSearchCV_tune():
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")
    
    train_csv, test_csv, encoder = lable_encoding(train_csv, test_csv)
    X_train, X_test, y_train, y_test = get_data(train_csv)
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    
    clf = get_xgboost(params={}, patience=PATIENCE, iterations=ITERATIONS)
    
    # Hyperparameter Optimization
    gsc = GridSearchCV(clf, param_grid= {} , cv=kf, verbose=100, refit=True)
    gsc.fit(X_train, y_train, early_stopping_rounds = PATIENCE) 
    x_predictsion = gsc.best_estimator_.predict(X_test)
    
    best_acc_score = accuracy_score(y_test, x_predictsion) 
    print(
    f"""
    {__name__}
    ============================================
    [best_acc_score : {best_acc_score}]
    [Best params : {gsc.best_params_}]
    [Best value: {gsc.best_score_}]
    ============================================
    """
    )

    # predict
    predictions = encoder.inverse_transform(gsc.best_estimator_.predict(test_csv)) 
    save(path, predictions)

#====================================================================================

global SEED, PATIENCE, ITERATIONS
SEED = 42
PATIENCE = 50
ITERATIONS = 1000
N_TRIALS = 10

#====================================================================================

# RUN
def main():
    obtuna_tune()
    # GridSearchCV_tune()

if __name__ == '__main__':
    main()