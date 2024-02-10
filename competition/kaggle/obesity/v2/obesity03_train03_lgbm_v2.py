# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from obesity01_data_v2 import lable_encoding, get_data, scaling, preprocessing, y_encoding
from obesity02_models_v2 import get_lightgbm, get_fitted_lightgbm
from obesity04_utils_v2 import save_csv, save_model
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
    
        #preprocessing
    train_csv = preprocessing(train_csv)
    test_csv = preprocessing(test_csv)
    
    #encoding
    train_csv["NObeyesdad"], inverse_dict = y_encoding(train_csv["NObeyesdad"])
    
    #to category -범주형 처리
    categirical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CALC", "MTRANS", "CAEC", "SCC" ,"SMOKE"]
    for column in categirical_columns :
        train_csv[column] = train_csv[column].astype('category')
        test_csv[column] = test_csv[column].astype('category')
    
    #slpit
    X_train, X_test, y_train, y_test = get_data(train_csv)
    
    #scaling
    numeric_colums = ["Age","Height","Weight","FCVC","NCP","CH2O","FAF","TUE", 'Meal_Habits','BMI'] 
    for column in numeric_colums:
        X_train[column], scaler = scaling(None, X_train[column].values.reshape(-1,1))
        X_test[column],_ = scaling(scaler, X_test[column].values.reshape(-1,1))
        test_csv[column],_ = scaling(scaler, test_csv[column].values.reshape(-1,1))  
    
    # Hyperparameter Optimization
    def objective(trial: optuna.Trial):
        params = {
            'learning_rate' : trial.suggest_float('learning_rate', 1e-3, 1e-1, log = True),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            # 'subsample' : trial.suggest_float('subsample', .5, 1),
            'min_child_weight' : trial.suggest_int('min_child_weight', 1, 20),
            'reg_lambda' : trial.suggest_float('reg_lambda',  1e-9, 1.0, log = True),
            'reg_alpha' : trial.suggest_float('reg_alpha',  1e-9, 1.0, log = True),
            'n_estimators' : iterations,
            'random_state' : SEED,
            'boosting_type' : 'gbdt',
            'imporance_type' : "feature_importances_",
            'device_type' : "gpu",
            # "objective": "multiclass" ,
            # "metric": 'cross_entropy',
            'num_leaves': trial.suggest_int('num_leaves', 10, 1000),  
            'verbosity' : -1        
        }
        #kfold 적용
        # acc_scores = np.empty(n_splits)
        # folds = StratifiedKFold(n_splits=n_splits, random_state=SEED, shuffle=True)
        # for idx, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
        #     X_train_, y_train_ = X_train.iloc[train_idx], y_train.iloc[train_idx]
        #     X_val_, y_val_ = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
        #     clf = get_fitted_lightgbm(params, X_train_, X_val_, y_train_, y_val_)
        #     predictions = clf.predict(X_test)
        #     acc_scores[idx] = accuracy_score(y_test, predictions)
        
        # print("Kfold mean acc: ", np.mean(acc_scores))
        # return np.mean(acc_scores)
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
    if gsc.best_score_ > 0.911:
        predictions = encoder.inverse_transform(gsc.best_estimator_.predict(test_csv)) 
        save_submit(path, round(gsc.best_score_,4), predictions)
# ====================================================================================

patience = 1
iterations = 1
n_trial = 1
n_splits = 2

# ====================================================================================

# RUN
def main():
    obtuna_tune()
    # GridSearchCV_tune()

if __name__ == '__main__':
    main()


#Trial 41 finished with value: 0.9836223506743738 and 
# parameters: {'learning_rate': 0.09984796553188138, 'max_depth': 5, 'subsample': 0.5429084022438241, 
# 'min_child_weight': 10.964349752559462, 'reg_lambda': 0.00021346694796729772, 'reg_alpha': 0.0002998057305075806, 'num_leaves': 219}.
    