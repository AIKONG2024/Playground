# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
from sklearn.metrics import accuracy_score
from obesity01_data import lable_encoding, get_data, y_encoding, x_preprocessing, train_only_preprocessing, get_data_val
from obesity02_models import get_randomForest, get_fitted_randomForest, get_catboost, get_fitted_catboost, get_xgboost, get_fitted_xgboost
from obesity04_utils import save_model,save_submit, save_csv
from obesity00_constant import SEED, ITERATTIONS, PATIENCE, N_TRIAL, N_SPLIT
# ====================================================================================
# test
def test():
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
    # train_csv = train_csv[train_csv['Age'] < 46]
    
    train_csv = pd.concat([train_csv.drop(categorical_features, axis=1), train_encoded_df], axis=1)
    test_csv = pd.concat([test_csv.drop(categorical_features, axis=1), test_encoded_df], axis=1)
    
    train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height'] ** 2)
    test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height'] ** 2)
    
    train_csv['Meal_Habits'] = train_csv['FCVC'] * train_csv['NCP']
    test_csv['Meal_Habits'] = test_csv['FCVC'] * test_csv['NCP']

    train_csv['Healthy_Nutrition_Habits'] = train_csv['FCVC'] / ( 2 * train_csv['FAVC_no'] - 1)
    test_csv['Healthy_Nutrition_Habits'] = test_csv['FCVC'] / ( 2 * test_csv['FAVC_no'] - 1)

    train_csv['Tech_Usage_Score'] = train_csv['TUE'] / train_csv['Age']
    test_csv['Tech_Usage_Score'] = test_csv['TUE'] / test_csv['Age']
    
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
    
    X_train, X_test, X_val, y_train, y_test, y_val = get_data_val(train_csv)

    xgb = get_xgboost(
        params={
            "grow_policy": "lossguide",
            "n_estimators": 900,
            "learning_rate": 0.029887991326059394,
            "gamma": 0.42260612306158285,
            "subsample": 0.5537373028286633,
            "colsample_bytree": 0.3353153186662535,
            "max_depth": 12,
            "min_child_weight": 7,
            "reg_lambda": 0.034523450966574416,
            "reg_alpha": 0.05405740444844809,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            'early_stopping_rounds' : 2000,
        }
    )
    # print(X_train.shape, y_train.shape)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)],verbose=False)
    print(train_csv.columns)
    print(xgb.feature_importances_)

    x_predictsion = xgb.predict(X_test)
    best_acc_score = accuracy_score(y_test, x_predictsion) 
    print(
    f"""
    {__name__}
    ============================================
    [best_acc_score : {best_acc_score}]
    ============================================
    """
    )

# ====================================================================================

patience = PATIENCE
iterations = ITERATTIONS
n_trial = N_TRIAL
n_splits = N_SPLIT

# ====================================================================================

# RUN
def main():
    # obtuna_tune()
    test()

if __name__ == '__main__':
    main()

# categorical_features = train_csv.columns[train_csv.dtypes=="object"].tolist()[:-1]
# from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder(sparse=False)
# encoder.fit(pd.concat([train_csv[categorical_features], test_csv[categorical_features]], axis=0))

# train_encoded = encoder.transform(train_csv[categorical_features])
# train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_features))
# test_encoded = encoder.fit_transform(test_csv[categorical_features])
# test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_features))
# combine_columns = ['CALC_Always', 'CALC_Frequently']
# train_encoded_df['CALC_A_F'] = train_encoded_df[combine_columns].sum(axis=1)
# test_encoded_df['CALC_A_F'] = test_encoded_df[combine_columns].sum(axis=1)

# train_encoded_df = train_encoded_df.drop(columns=combine_columns).set_index(train_csv.index)
# test_encoded_df = test_encoded_df.drop(columns=combine_columns).set_index(test_csv.index)

# # levels = {"Always": 3, "Frequently": 2, "Sometimes": 1, "no": 0}
# # train_csv["CALC_ord"] = train_csv["CALC"].map(levels)
# # test_csv["CALC_ord"] = test_csv["CALC"].map(levels)
# # train_csv["CAEC_ord"] = train_csv["CAEC"].map(levels)
# # test_csv["CAEC_ord"] = test_csv["CAEC"].map(levels)
# # train_csv = train_csv[train_csv['Age'] < 46]

# train_csv = pd.concat([train_csv.drop(categorical_features, axis=1), train_encoded_df], axis=1)
# test_csv = pd.concat([test_csv.drop(categorical_features, axis=1), test_encoded_df], axis=1)

# train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height'] ** 2)
# test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height'] ** 2)

# train_csv['Meal_Habits'] = train_csv['FCVC'] * train_csv['NCP']
# test_csv['Meal_Habits'] = test_csv['FCVC'] * test_csv['NCP']

# train_csv['Healthy_Nutrition_Habits'] = train_csv['FCVC'] / ( 2 * train_csv['FAVC_no'] - 1)
# test_csv['Healthy_Nutrition_Habits'] = test_csv['FCVC'] / ( 2 * test_csv['FAVC_no'] - 1)

# train_csv['Tech_Usage_Score'] = train_csv['TUE'] / train_csv['Age']
# test_csv['Tech_Usage_Score'] = test_csv['TUE'] / test_csv['Age']

# cat_features = train_csv.select_dtypes(include='object').columns.values
# for feature in cat_features :
#     train_csv[feature], lbe = lable_encoding(None,train_csv[feature]) 
#     if feature != "NObeyesdad":
#         test_csv[feature],_ = lable_encoding(lbe, test_csv[feature]) 

# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # sns.set(font_scale=0.7)
# # sns.heatmap(data=train_csv.corr(), square=True, annot=True, cbar=True) 
# # plt.show()

# X_train, X_test, X_val, y_train, y_test, y_val = get_data_val(train_csv)
# "grow_policy": "lossguide",
# "n_estimators": 900,
# "learning_rate": 0.029887991326059394,
# "gamma": 0.42260612306158285,
# "subsample": 0.5537373028286633,
# "colsample_bytree": 0.3353153186662535,
# "max_depth": 12,
# "min_child_weight": 7,
# "reg_lambda": 0.024523450966574416,
# "reg_alpha": 0.02405740444844809,
# "objective": "multi:softprob",
# "eval_metric": "mlogloss",
# 'early_stopping_rounds' : 1000,


# params={
#     "grow_policy": "lossguide",
#     "n_estimators": 900,
#     "learning_rate": 0.029887991326059394,
#     "gamma": 0.42260612306158285,
#     "subsample": 0.5537373028286633,
#     "colsample_bytree": 0.3353153186662535,
#     "max_depth": 12,
#     "min_child_weight": 7,
#     "reg_lambda": 0.034523450966574416,
#     "reg_alpha": 0.05405740444844809,
#     "objective": "multi:softprob",
#     "eval_metric": "mlogloss",
#     'early_stopping_rounds' : 1000,
# }