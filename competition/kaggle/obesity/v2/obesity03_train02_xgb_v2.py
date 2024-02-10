# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from obesity01_data_v2 import lable_encoding, get_data, scaling
from obesity02_models_v2 import get_xgboost, get_fitted_xgboost
from obesity04_utils_v2 import save_submit, save_model
from obesity00_seed_v2 import SEED
from sklearn.model_selection import StratifiedKFold

#====================================================================================
#obtuna Tunner 이용
def obtuna_tune():
    import optuna
    
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")
    #encoding
    train_csv["NObeyesdad"], lbe = lable_encoding(None, train_csv["NObeyesdad"])
    
    #to category -범주형 처리
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
        

    def objective(trial: optuna.Trial):
        params = {
            'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 3.0),
            'gamma' : trial.suggest_float('gamma', 1e-9, 1.0), #필수
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1.0, log=True),   
            # 'scale_pos_weight' :  trial.suggest_int('scale_pos_weight', 1, 10, log=True), 
            # 'subsample': trial.suggest_float('subsample', 0.25, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.25, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 6),#필수
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20), #필수 
            'eval_metric' : 'auc',
            'booster' : trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'verbosity' : 1,
            'objective' : trial.suggest_categorical('objective', ["multi:softprob", "multi:softmax"]),
            # 'task_type': 'GPU',
            'device' : 'cuda',
            'enable_categorical' : True,
            # 'max_cat_to_onehot' : 7,
            'early_stopping_rounds' : patience,
            # 'importance_type' : 'weight',
            'random_state' : SEED,
        }
        #kfold 적용
        acc_scores = np.empty(n_splits)
        folds = StratifiedKFold(n_splits=n_splits, random_state=SEED, shuffle=True)
        for idx, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
            X_train_, y_train_ = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val_, y_val_ = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
            clf = get_fitted_xgboost(params, X_train_, X_val_, y_train_, y_val_)
            predictions = clf.predict(X_test)
            acc_scores[idx] = accuracy_score(y_test, predictions)
        
        print("Kfold mean acc: ", np.mean(acc_scores))
        return np.mean(acc_scores)
        
        # clf = get_fitted_xgboost(params, X_train, X_test, y_train, y_test)
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
    if "enable_categorical" not in best_study.params :
        best_study.params['enable_categorical'] = True
    best_model = get_fitted_xgboost(best_study.params, X_train, X_test, y_train, y_test)  # bestest
    predictions = lbe.inverse_transform(best_model.predict(test_csv))

    if best_study.value > 0.911:
        save_submit(path, f"{round(best_study.value,4)}_xboost", predictions)
        save_model(path, f"{round(best_study.value,4)}_xboost", best_model)

    
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
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    clf = get_xgboost(params= {})
    
    # Hyperparameter Optimization
    gsc = GridSearchCV(clf, param_grid={
            'gamma' :  [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0], #필수
            'max_depth': [0, 4,8,12,16,20, 24],#필수
            'min_child_weight': [1,3,5,7,9,12,15,18,21,24,27, 30], #필수 
            'eval_metric' : ['auc'],
            'booster' : ['gbtree'],
            'verbosity' : [0],
            'objective' : ["multi:sotfmax"],
            'device' : ['cuda'],
            'enable_categorical' : [True],
            'max_cat_to_onehot' : [7],
            'early_stopping_rounds' :[ patience],
            'importance_type' : ['weight'],
            'random_state' : [SEED],
        } , cv=kf, verbose=100, refit=True)
    gsc.fit(X_train, y_train, eval_set=[(X_test, y_test)] ,verbose=False)
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
#====================================================================================

patience = 10
iterations = 300
n_trial = 50
n_splits = 6

#====================================================================================

# RUN
def main():
    obtuna_tune()

if __name__ == '__main__':
    main()
    
# Hyperparameter Optimization
# Trial 8 finished with value: 0.9132947976878613 and 
# parameters: {'grow_policy': 'depthwise', 'n_estimators': 835, 'learning_rate': 0.4590126797048605, 
# 'gamma': 0.697583795378372, 'reg_lambda': 2.5571626843725646e-09, 'reg_alpha': 0.025830165683120763,
# 'colsample_bytree': 0.6839781896116557, 'max_depth': 6, 'min_child_weight': 12, 'booster': 'gbtree'}. 
# Best is trial 8 with value: 0.9132947976878613.

#Trial 22 finished with value: 0.9137764932562621 and parameters: {'grow_policy': 'lossguide', 'n_estimators': 869, 
# 'learning_rate': 0.3028652380564825, 'gamma': 0.2856670536033816, 'reg_lambda': 4.9680675237488436e-08, 
# 'reg_alpha': 0.00016359246091733195, 'colsample_bytree': 0.5084997693262157, 'max_depth': 4,
# 'min_child_weight': 1, 'booster': 'gbtree'}. Best is trial 22 with value: 0.9137764932562621.


#Trial 38 finished with value: 0.9137764932562621 and parameters: {'grow_policy': 'lossguide', 'n_estimators': 278, 
# 'learning_rate': 0.3066571522862226, 'gamma': 0.2763507041041538, 'reg_lambda': 0.11446738122556356, 'reg_alpha': 1.567883180948874e-05, 
# 'colsample_bytree': 0.48985814279942413, 'max_depth': 5, 'min_child_weight': 13, 'booster': 'dart', 'objective': 'multi:softmax'}.
# Best is trial 38 with value: 0.9137764932562621.

'''
분석 : 
learning_rate : 0.3028652, 0.30665715 , 
grow_policy : lossguide, depthwise
n_estimate : x
gamma : 0.28,0.27
lambda : x
alpha : x
colsample : 0.4~0.9
boost : bart, gbtree,
objective : x


'''
