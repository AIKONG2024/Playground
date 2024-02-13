# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
from sklearn.metrics import accuracy_score
from obesity01_data import lable_encoding, get_data, y_encoding, x_preprocessing, train_only_preprocessing, scaling
from obesity02_models import get_catboost, get_fitted_catboost
from obesity04_utils import save_submit, save_model, save_csv
from obesity00_constant import SEED, ITERATTIONS, PATIENCE, N_TRIAL, N_SPLIT

# ====================================================================================
# obtuna Tunner 이용
def obtuna_tune():
    import optuna
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")

    train_csv = train_only_preprocessing(train_csv)
    train_csv = x_preprocessing(train_csv)
    test_csv =  x_preprocessing(test_csv)
    
    scaling_features = ["Weight", "Height"]
    for feature in scaling_features :
        train_csv[feature], scaler = scaling(None, train_csv[feature])
        test_csv[feature], _ = scaling(scaler, test_csv[feature])
        
    cat_features = train_csv.select_dtypes(include='object').columns.values[:-1]
    train_csv["NObeyesdad"], inverse_dict = y_encoding(train_csv["NObeyesdad"])
    
    
    X_train, X_test, y_train, y_test = get_data(train_csv)
    # Hyperparameter Optimization
    def objective(trial: optuna.Trial):
        params = {
            'iterations': iterations,  # High number of estimators
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
            # 'depth': trial.suggest_int('depth', 3, 15),
            # 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 30.0),
            # 'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_seed': SEED,
            'verbose': False,
            'task_type':"GPU"          
        }
        clf = get_fitted_catboost(params, X_train, X_test, y_train, y_test, cat_features)

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
    if best_study.value > 0.91:
        best_model = get_fitted_catboost(best_study.params, X_train, X_test, y_train, y_test, cat_features)  # bestest
        predictions = best_model.predict(test_csv)[:, 0]
        submission_csv = pd.read_csv(path + "sample_submission.csv")
        submission_csv["NObeyesdad"] = predictions
        submission_csv["NObeyesdad"] = submission_csv["NObeyesdad"].map(inverse_dict)
        save_csv(path, f"{round(best_study.value,4)}_catboost_", submission_csv)
        save_model(path, f"{round(best_study.value,4)}_catboost_", best_model)


# ====================================================================================
# GridSearchCV Tunner 이용
def GridSearchCV_tune():
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")
    
    train_csv, test_csv, encoder = lable_encoding(train_csv, test_csv)
    X_train, X_test, y_train, y_test = get_data(train_csv)
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    clf = get_catboost(params ={})
    
    # Hyperparameter Optimization
    gsc = GridSearchCV(clf, param_grid=  {
            'iterations': [iterations],  
            'learning_rate': [0.01, 0.3],
            'depth':  [3, 10],
            'l2_leaf_reg':  [0.01, 10.0],
            'bagging_temperature':  [0.0, 1.0],
            'random_seed': [SEED],
            'verbose': [True],
            'task_type':["GPU"]
            })
    
    gsc.fit(X_train, y_train, early_stopping_rounds = patience) 
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
