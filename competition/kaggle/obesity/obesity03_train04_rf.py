# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
from sklearn.metrics import accuracy_score
from obesity01_data import lable_encoding, get_data
from obesity02_models import get_randomForest, get_fitted_randomForest
from obesity04_utils import save
from obesity00_seed import SEED

# ====================================================================================
# obtuna Tunner 이용
def obtuna_tune():
    import optuna
    
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")
    
    train_csv, test_csv, encoder = lable_encoding(train_csv, test_csv)
    datasets = get_data(train_csv)

    # Hyperparameter Optimization
    def objective(trial: optuna.Trial):
        params = {
            'max_depth' : trial.suggest_int('max_depth', 2, 20),
            'min_samples_split' : trial.suggest_float('min_samples_split', 1, 4),
            'min_samples_leaf' : trial.suggest_float('min_samples_leaf', .1, 50),
            'max_depth' : trial.suggest_int('max_depth', 1, 30),
            'max_samples' : trial.suggest_float('max_samples', 0, 1.),
            'n_estimators' : iterations,
            'random_state' : SEED,
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),          
        }
        clf = get_fitted_randomForest(params, datasets)
        
        X_test, y_test = datasets[1], datasets[3]
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
    best_model = get_fitted_randomForest(best_study.params, datasets)  # bestest
    predictions = encoder.inverse_transform(best_model.predict(test_csv))
    save(path, round(best_study.value,4), predictions)


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
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    clf = get_randomForest(params={})

    # Hyperparameter Optimization
    gsc = GridSearchCV(clf, param_grid= {
            'max_depth' : [2, 20],
            'min_samples_split' : [.5, 1],
            'min_samples_leaf' : [.1, 15],
            'max_depth' : [1, 20],
            'max_samples' : [0.1, 0.5],
            'n_estimators' : [iterations],
            'random_state' : [SEED],
            'max_leaf_nodes':[10, 20],          
        } , cv=kf, verbose=100, refit=True)
    gsc.fit(X_train, y_train) 
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
    save(path, round(gsc.best_score_,4), predictions)

# ====================================================================================

patience = 1000
iterations = 3000
n_trial = 5
n_splits = 5

# ====================================================================================

# RUN
def main():
    obtuna_tune()
    # GridSearchCV_tune()

if __name__ == '__main__':
    main()
