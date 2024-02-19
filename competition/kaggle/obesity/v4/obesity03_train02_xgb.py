# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from obesity01_data import lable_encoding, get_data, y_encoding, x_preprocessing, train_only_preprocessing
from obesity02_models import get_xgboost, get_fitted_xgboost
from obesity04_utils import save_submit, save_model, save_csv
from obesity00_constant import SEED, ITERATTIONS, PATIENCE, N_TRIAL, N_SPLIT

#====================================================================================
#obtuna Tunner 이용
def obtuna_tune():
    import optuna
    
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")
    
    categorical_features = train_csv.columns[train_csv.dtypes=="object"].tolist()[:-1]
    from sklearn.preprocessing import OneHotEncoder
    
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(pd.concat([train_csv[categorical_features], test_csv[categorical_features]], axis=0))

    train_encoded = encoder.transform(train_csv[categorical_features])
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_features))
    test_encoded = encoder.fit_transform(test_csv[categorical_features])
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_features))
    combine_columns = ['CALC_Always', 'CALC_Frequently']
    train_encoded_df['CALC_A_F'] = train_encoded_df[combine_columns].sum(axis=1)
    test_encoded_df['CALC_A_F'] = test_encoded_df[combine_columns].sum(axis=1)

    train_encoded_df = train_encoded_df.drop(columns=combine_columns).set_index(train_csv.index)
    test_encoded_df = test_encoded_df.drop(columns=combine_columns).set_index(test_csv.index)
    
    # levels = {"Always": 3, "Frequently": 2, "Sometimes": 1, "no": 0}
    # train_csv["CALC_ord"] = train_csv["CALC"].map(levels)
    # test_csv["CALC_ord"] = test_csv["CALC"].map(levels)
    # train_csv["CAEC_ord"] = train_csv["CAEC"].map(levels)
    # test_csv["CAEC_ord"] = test_csv["CAEC"].map(levels)
    train_csv = train_csv[train_csv['Age'] > 20]
    train_csv = pd.concat([train_csv.drop(categorical_features, axis=1), train_encoded_df], axis=1)
    test_csv = pd.concat([test_csv.drop(categorical_features, axis=1), test_encoded_df], axis=1)
    
    train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height'] ** 2)
    test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height'] ** 2)
    
    # train_csv['Meal_Habits'] = train_csv['FCVC'] * train_csv['NCP']
    # test_csv['Meal_Habits'] = test_csv['FCVC'] * test_csv['NCP']

    # train_csv['Healthy_Nutrition_Habits'] = train_csv['FCVC'] / ( 2 * train_csv['FAVC_no'] - 1)
    # test_csv['Healthy_Nutrition_Habits'] = test_csv['FCVC'] / ( 2 * test_csv['FAVC_no'] - 1)

    # train_csv['Tech_Usage_Score'] = train_csv['TUE'] / train_csv['Age']
    # test_csv['Tech_Usage_Score'] = test_csv['TUE'] / test_csv['Age']
    
    cat_features = train_csv.select_dtypes(include='object').columns.values
    for feature in cat_features :
        train_csv[feature], lbe = lable_encoding(None,train_csv[feature]) 
        if feature != "NObeyesdad":
            test_csv[feature],_ = lable_encoding(lbe, test_csv[feature]) 
    
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set(font_scale=0.7)
    # sns.heatmap(data=train_csv.corr(), square=True, annot=True, cbar=True) 
    # plt.show()
    
    X_train, X_test, y_train, y_test = get_data(train_csv)

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
            'device_type' : 'GPU',
            'device' : 'cuda',
            'tree_method' : 'hist',
            # 'enable_categorical' : True,
            # 'max_cat_to_onehot' : 1,
            'early_stopping_rounds' : patience,
            # 'importance_type' : 'weight',
            'random_state' : SEED,
        }
        
        clf = get_fitted_xgboost(params, X_train, X_test, y_train, y_test,)
        
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
    best_model = get_fitted_xgboost(best_study.params, X_train, X_test, y_train, y_test)  # bestest
    # print(best_model.feature_importances_)
    predictions = best_model.predict(test_csv)
    submission_csv = pd.read_csv(path + "sample_submission.csv")
    submission_csv["NObeyesdad"] = lbe.inverse_transform(predictions) 
    # submission_csv["NObeyesdad"] = predictions
    # submission_csv["NObeyesdad"] = submission_csv["NObeyesdad"].map(inverse_dict)
    
    save_csv(path, f"{round(best_study.value,4)}_xgb_", submission_csv)
    save_model(path, f"{round(best_study.value,4)}_xgb_", best_model)

    
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

patience = PATIENCE
iterations = ITERATTIONS
n_trial = N_TRIAL
n_splits = N_SPLIT

#====================================================================================

# RUN
def main():
    obtuna_tune()
    # GridSearchCV_tune()

if __name__ == '__main__':
    main()
    
# SEED = 42
# PATIENCE = 300
# ITERATTIONS = 500
# N_TRIAL = 100
# N_SPLIT = 5
    # Trial 12 finished with value: 0.9150610147719974 and parameters: {'grow_policy': 'lossguide', 'n_estimators': 996, 'learning_rate': 0.022115905415387556, 'gamma': 0.1299998264675535, 'subsample': 0.7591158724763973, 
    # 'colsample_bytree': 0.3015166580857436, 'max_depth': 7, 'min_child_weight': 7, 'reg_lambda': 0.035320295955169626, 'reg_alpha': 1.645889988069947, 'objective': 'multi:softprob', 'eval_metric': 'auc'}. Best is trial 12 with value: 0.9150610147719974.
    
    # Trial 51 finished with value: 0.9153821451509313 and parameters: {'grow_policy': 'lossguide', 'n_estimators': 714, 'learning_rate': 0.017326110064790005, 'gamma': 0.4034388490359524, 'subsample': 0.6733880151714363, 
    # 'colsample_bytree': 0.30100800155682894, 'max_depth': 10, 'min_child_weight': 7, 'reg_lambda': 0.43190307429710767, 'reg_alpha': 0.10634051992652219, 'objective': 'multi:sotfmax', 'eval_metric': 'merror'}. Best is trial 51 with value: 0.9153821451509313.
   
    #  Trial 69 finished with value: 0.9169877970456005 and parameters: {'grow_policy': 'lossguide', 'n_estimators': 813, 'learning_rate': 0.029887991326059394, 'gamma': 0.42260612306158285, 'subsample': 0.5537373028286633, 
    # 'colsample_bytree': 0.3353153186662535, 'max_depth': 12, 'min_child_weight': 7, 'reg_lambda': 0.024523450966574416, 'reg_alpha': 0.02405740444844809, 'objective': 'multi:sotfmax', 'eval_metric': 'merror'}. Best is trial 69 with value: 0.9169877970456005.
    
    
# SEED = 42
# PATIENCE = 300
# ITERATTIONS = 500
# N_TRIAL = 200
# N_SPLIT = 5

