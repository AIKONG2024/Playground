# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
from sklearn.metrics import accuracy_score
from obesity01_data import lable_encoding, get_data
from obesity02_models import get_catboost, get_fitted_catboost
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

    cat_features = train_csv.select_dtypes(include='object').columns.values[:-1]
    datasets = get_data(train_csv)

    # Hyperparameter Optimization
    def objective(trial: optuna.Trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),            
        }
        clf = get_fitted_catboost(params, datasets, PATIENCE, ITERATIONS, cat_features)

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
    best_model = get_fitted_catboost(best_study.params, datasets, cat_features)  # bestest
    predictions = best_model.predict(test_csv)[:, 0]
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
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    clf = get_catboost(params = {"learning_rate" : [1e-3, 1e-2, 1e-1] }, patience=PATIENCE, iterations=ITERATIONS)
    
    # Hyperparameter Optimization
    gsc = GridSearchCV(clf, param_grid=  {"learning_rate" : [1e-3, 1e-2, 1e-1] } , cv=kf, verbose=100, refit=True)
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
    save(path, round(gsc.best_score_,4), predictions)

# ====================================================================================

global PATIENCE, ITERATIONS, N_TRIALS
PATIENCE = 100
ITERATIONS = 1000
N_TRIALS = 10
n_splits = 5

# ====================================================================================

# RUN
def main():
    # obtuna_tune()
    GridSearchCV_tune()

if __name__ == '__main__':
    main()
