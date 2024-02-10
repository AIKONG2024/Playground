# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from obesity01_data_v2 import lable_encoding, get_data, scaling, onehot_encoding, y_encoding,preprocessing
from obesity02_models_v2 import get_randomForest, get_fitted_randomForest
from obesity04_utils_v2 import save_model,save_csv
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
    
    train_csv = preprocessing(train_csv)
    test_csv = preprocessing(test_csv)
    
    #encoding
    categirical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
    for column in categirical_columns:
        train_csv[column] , ohe = onehot_encoding(None, train_csv[column])
        test_csv[column], _ = onehot_encoding(ohe, test_csv[column]) 
    train_csv["NObeyesdad"], inverse_dict = y_encoding(train_csv["NObeyesdad"])
    
    categirical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
    for column in categirical_columns :
        train_csv[column] = train_csv[column].astype('category')
        test_csv[column] = test_csv[column].astype('category')
    
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
            'max_depth' : trial.suggest_int('max_depth', 1, 20),
            'n_estimators' : iterations,
            'criterion' : trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'class_weight' : 'balanced',
            'random_state' : SEED,
            'bootstrap' : True,
            # 'feature_names_in' : True
        }
        #kfold 적용
        # acc_scores = np.empty(n_splits)
        # folds = StratifiedKFold(n_splits=n_splits, random_state=SEED, shuffle=True)
        # for idx, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
        #     X_train_, y_train_ = X_train.iloc[train_idx], y_train.iloc[train_idx]
        #     X_val_, y_val_ = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
        #     clf = get_fitted_randomForest(params, X_train_, X_val_, y_train_, y_val_)
        #     predictions = clf.predict(X_test)
        #     acc_scores[idx] = accuracy_score(y_test, predictions)
        
        # print("Kfold mean acc: ", np.mean(acc_scores))
        # return np.mean(acc_scores)
        clf = get_fitted_randomForest(params, X_train, X_test, y_train, y_test)
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
    best_model = get_fitted_randomForest(best_study.params, X_train, X_test, y_train, y_test)  # bestest
    predictions = best_model.predict(test_csv)
    if best_study.value > 0.911:
        submission_csv = pd.read_csv(path + "sample_submission.csv")
        submission_csv["NObeyesdad"] = predictions
        submission_csv["NObeyesdad"] = submission_csv["NObeyesdad"].map(inverse_dict)
        save_csv(path, f"{round(best_study.value,4)}_rf_", submission_csv)
        save_model(path, f"{round(best_study.value,4)}_tf_", best_model)


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
    save_submit(path, round(gsc.best_score_,4), predictions)

# ====================================================================================

patience = 1
iterations = 1
n_trial = 1
n_splits = 6

# ====================================================================================

# RUN
def main():
    obtuna_tune()
    # GridSearchCV_tune()

if __name__ == '__main__':
    main()
