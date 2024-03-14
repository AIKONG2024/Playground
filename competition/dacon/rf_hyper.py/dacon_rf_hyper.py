import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score 
import optuna
from collections import OrderedDict

path = 'C:/_data/dacon/rf_hyper/'
SEED = 42
train_csv = pd.read_csv(path + "train.csv", index_col=0)
X = train_csv.drop('login', axis=1)
y = train_csv['login']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
skf = StratifiedKFold(n_splits=4)

for train, test in skf.split(X_train, y_train):
    # print('train -  {}   |   test -  {}'.format(
    #     np.bincount(y[train]), np.bincount(y[test])))
    X_train = X_train[train]
    y_train = y_train[train]
    X_test = X_train[test]
    y_test = y_train[test]
    #def : 8595
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'criterion': trial.suggest_categorical('criterion', ['gini','entropy']),
            'max_depth': trial.suggest_int('min_samples_split', 2, 300),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
            'min_weight_fraction_leaf': 0.0,
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'max_leaf_nodes': None, # Example alternative
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': SEED
        }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return roc_auc_score(y, predictions)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=400)
    best_params = study.best_trial.params

    # best_params = {'n_estimators': 37, 'max_depth' : 100, 'min_weight_fraction_leaf' : 0.0 , 'max_leaf_nodes': None, 'criterion': 'entropy', 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': True}
    # best_params = {'n_estimators': 37, 'criterion': 'entropy', 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': True}
    print(best_params)
    model = RandomForestClassifier(**best_params, random_state=SEED)
    model.fit(X_test, y_test)
    predictions = model.predict(X)
    score = roc_auc_score(y, predictions)
    print("ROC AUC score : ", score)

    param_order = [
        'n_estimators',
        'criterion',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'min_weight_fraction_leaf',
        'max_features',
        'max_leaf_nodes',
        'min_impurity_decrease',
        'bootstrap',
    ]
    best_params_ordered = OrderedDict({k: best_params.get(k, None) for k in param_order})

    best_params_ordered['max_depth'] = 100  
    best_params_ordered['min_weight_fraction_leaf'] = 0.0
    best_params_ordered['max_leaf_nodes'] = None  

    if score > 0.88:
        submission = pd.DataFrame([best_params_ordered])
        submission.to_csv(path + f'sample_submission_pred{score}.csv', index=False)
