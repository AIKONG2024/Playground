# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from obesity01_data_v2 import lable_encoding, get_data, scaling
from obesity02_models_v2 import get_catboost, get_fitted_catboost
from obesity04_utils_v2 import save_model, save_submit
from obesity00_seed_v2 import SEED
from sklearn.model_selection import StratifiedKFold

# ====================================================================================
# obtuna Tunner 이용
def obtuna_tune():
    import optuna
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")

    #encoding
    categirical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
    # for column in categirical_columns:
    #     train_csv[column] , ohe = onehot_encoding(None, train_csv[column])
    #     test_csv[column], _ = onehot_encoding(ohe, test_csv[column]) 
    train_csv["NObeyesdad"], lbe = lable_encoding(None, train_csv["NObeyesdad"])
    
    # categirical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
    # for column in categirical_columns :
    #     train_csv[column] = train_csv[column].astype('category')
    #     test_csv[column] = test_csv[column].astype('category')
    
    #slpit
    X_train, X_test, y_train, y_test = get_data(train_csv)
    
    #scaling
    numeric_colums = ["Age","Height","Weight","FCVC","NCP","CH2O","FAF","TUE"] 
    for column in numeric_colums:
        X_train[column], scaler = scaling(None, X_train[column].values.reshape(-1,1))
        X_test[column],_ = scaling(scaler, X_test[column].values.reshape(-1,1))
        test_csv[column],_ = scaling(scaler, test_csv[column].values.reshape(-1,1))

    # Hyperparameter Optimization
    def objective(trial: optuna.Trial):
        params = {
            'iterations': iterations, 
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1),
            'depth': trial.suggest_int('depth', 2, 6),
            # 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 30.0),
            # 'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_seed': SEED,
            # 'verbose': False,
            'task_type':"GPU"          
        }
        #kfold 적용
        acc_scores = np.empty(n_splits)
        folds = StratifiedKFold(n_splits=n_splits, random_state=SEED, shuffle=True)
        for idx, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
            X_train_, y_train_ = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val_, y_val_ = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
            clf = get_fitted_catboost(params, X_train_, X_val_, y_train_, y_val_, categirical_columns)
            predictions = clf.predict(X_test)
            acc_scores[idx] = accuracy_score(y_test, predictions)
        
        print("Kfold mean acc: ", np.mean(acc_scores))
        return np.mean(acc_scores)
        
        # clf = get_fitted_catboost(params, X_train, X_test, y_train, y_test, categirical_columns)

        # predictions = clf.predict(X_test)
        # return accuracy_score(y_test, predictions)

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
    if best_study.value > 0.911:
        best_model = get_fitted_catboost(best_study.params, X_train, X_test, y_train, y_test, categirical_columns)  # bestest
        predictions = best_model.predict(test_csv)[:, 0]
        save_submit(path, f"{round(best_study.value,4)}_catboost", predictions)
        save_model(path, f"{round(best_study.value,4)}_catboost", best_model)


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
patience = 30
iterations = 300
n_trial = 50
n_splits = 6
# ====================================================================================

# RUN
def main():
    obtuna_tune()
    # GridSearchCV_tune()

if __name__ == '__main__':
    main()
