# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
from sklearn.metrics import accuracy_score
from obesity01_data import lable_encoding, get_data, y_encoding, x_preprocessing, train_only_preprocessing
from obesity02_models import get_lightgbm, get_fitted_lightgbm
from obesity04_utils import save_submit, save_model, save_csv
from obesity00_constant import SEED, ITERATTIONS, PATIENCE, N_TRIAL, N_SPLIT
import warnings

# ====================================================================================
# obtuna Tunner 이용
def obtuna_tune():
    import optuna
    
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")
    
    train_csv = train_only_preprocessing(train_csv)
    train_csv =  x_preprocessing(train_csv)
    test_csv =  x_preprocessing(test_csv)
    
    cat_features = train_csv.select_dtypes(include='object').columns.values[:-1]
    for feature in cat_features :
        train_csv[feature], lbe = lable_encoding(None,train_csv[feature]) 
        test_csv[feature],_ = lable_encoding(lbe, test_csv[feature]) 
    train_csv["NObeyesdad"], inverse_dict = y_encoding(train_csv["NObeyesdad"])
    
    X_train, X_test, y_train, y_test = get_data(train_csv)

    # Hyperparameter Optimization
    def objective(trial: optuna.Trial):
        params = {
            'learning_rate' : trial.suggest_float('learning_rate', .001, .1, log = True),
            'max_depth' : trial.suggest_int('max_depth', 2, 15),
            # 'subsample' : trial.suggest_float('subsample', .5, 1),
            'min_child_weight' : trial.suggest_float('min_child_weight', .1, 15, log = True),
            # 'reg_lambda' : trial.suggest_float('reg_lambda',  1e-9, 1.0, log = True),
            # 'reg_alpha' : trial.suggest_float('reg_alpha',  1e-9, 1.0, log = True),
            'n_estimators' : iterations,
            'random_state' : SEED,
            'device_type' : "gpu",
            # 'num_leaves': trial.suggest_int('num_leaves', 10, 1000),  
            'verbose' : -1        
        }
        clf = get_fitted_lightgbm(params, X_train, X_test, y_train, y_test)
        
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
    best_model = get_fitted_lightgbm(best_study.params, X_train, X_test, y_train, y_test)  # bestest
    predictions = best_model.predict(test_csv)
    if best_study.value > 0.91:
        submission_csv = pd.read_csv(path + "sample_submission.csv")
        submission_csv["NObeyesdad"] = predictions
        submission_csv["NObeyesdad"] = submission_csv["NObeyesdad"].map(inverse_dict)
        save_csv(path, f"{round(best_study.value,4)}_light_gbm", submission_csv)
        save_model(path, f"{round(best_study.value,4)}_light_gbm", best_model)

# ====================================================================================
# GridSearchCV Tunner 이용
def GridSearchCV_tune():
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")
    
    train_csv = train_csv[train_csv["Age"] < 46]

    train_csv, test_csv, encoder = lable_encoding(train_csv, test_csv)
    X_train, X_test, y_train, y_test = get_data(train_csv)
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    clf = get_lightgbm(params={})

    # Hyperparameter Optimization
    gsc = GridSearchCV(clf, param_grid= {
            'learning_rate' : [.001, .1],
            'max_depth' : [2, 15] ,
            'subsample' : [5, 1],
            'min_child_weight' : [ .1, 15],
            'reg_lambda' : [.1, 20],
            'reg_alpha' : [.1, 10],
            'n_estimators' : [iterations],
            'random_state' : [SEED],
            'device_type' : ["gpu"],
            'num_leaves': [10, 1000],    
        } , cv=kf, verbose=-1, refit=True)
    gsc.fit(X_train, y_train,eval_set=[(X_train, y_train),(X_test, y_test)])
    x_predictsion = gsc.best_estimator_.predict(X_test)

    best_acc_score = accuracy_score(y_test, x_predictsion) 
    print(
    f"""
    ============================================
    [best_acc_score : {best_acc_score}]
    [Best params : {gsc.best_params_}]
    [Best value: {gsc.best_score_}]
    ============================================
    """
    )

    # predict
    if gsc.best_score_ > 0.91:
        predictions = encoder.inverse_transform(gsc.best_estimator_.predict(test_csv)) 
        save_submit(path, round(gsc.best_score_,4), predictions)

# ====================================================================================

patience = PATIENCE
iterations = ITERATTIONS
n_trial = N_TRIAL
n_splits = N_SPLIT

# ====================================================================================

# RUN
def main():
    obtuna_tune()
    # GridSearchCV_tune()

if __name__ == '__main__':
    main()
