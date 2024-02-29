# https://www.kaggle.com/competitions/playground-series-s4e2
import pandas as pd
from sklearn.metrics import accuracy_score
from obesity02_models import get_randomForest, get_fitted_randomForest, get_xgboost
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from obesity00_constant import SEED, ITERATTIONS, PATIENCE, N_TRIAL, N_SPLIT
# ====================================================================================
#test
def test():
    import optuna
    
    # get data
    path = "C:/_data/kaggle/obesity/"
    train_csv = pd.read_csv(path + "train.csv")
    test_csv = pd.read_csv(path + "test.csv")
    
    train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height'] ** 2)
    test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height'] ** 2)
    
    lbe= LabelEncoder()
    train_gender = lbe.fit_transform(train_csv['Gender'])
    test_gender = lbe.transform(test_csv['Gender'])
    train_csv['BFP'] = (1.2*train_csv['BMI']) + (0.23*train_csv['Age']) - ((train_gender * 1) + (train_gender + 1)) * 5.4
    test_csv['BFP'] = (1.2*test_csv['BMI']) + (0.23*test_csv['Age']) - ((test_gender * 1) + (test_gender + 1)) * 5.4
        
    features = list(train_csv.columns)
    to_remove = ['id','SMOKE']
    [features.remove(feature) for feature in to_remove if feature in to_remove]
    
    cat_features = train_csv.select_dtypes(include='object').columns.values
    for feature in cat_features :
        train_csv[feature] = train_csv[feature].astype('category')
        if feature != 'NObeyesdad':
            test_csv[feature] = test_csv[feature].astype('category')
            
    train_csv['NObeyesdad'] = lbe.fit_transform(train_csv['NObeyesdad'])
    
    X, y = train_csv.drop(["NObeyesdad"], axis=1), train_csv.NObeyesdad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    params = {'n_estimators': 1500, 'max_depth': 4, 'min_samples_split': 1, 'min_samples_leaf': 1, 
                       'learning_rate': 0.3688172, 'gamma': 0.0542, 'random_state': 42, 'enable_categorical' : True}
    cls = XGBClassifier(**params)
    cls.fit(X_train, y_train, eval_set=[(X_test, y_test)],verbose=False)
    print(train_csv.columns)
    
    x_predictsion = cls.predict(X_test)
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
