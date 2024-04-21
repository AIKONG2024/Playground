import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score 
import optuna
from collections import OrderedDict

path = 'C:/_data/dacon/rf_hyper/'
SEED = 1234
train_csv = pd.read_csv(path + "train.csv", index_col=0)
X = train_csv.drop('login', axis=1)
y = train_csv['login']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=SEED)


#def : 8595
def objective(trial):
    params = {
        'n_estimators': 300,
        'criterion': trial.suggest_categorical('criterion', ['entropy']),
        'max_depth': 70,
        'min_samples_split': trial.suggest_int(name="min_samples_split", low=2, high=100, step=2),
        'min_samples_leaf': 4,
        'min_weight_fraction_leaf': 0.0,
        'max_features': trial.suggest_categorical(name="max_features", choices=[None]),
        'max_leaf_nodes': trial.suggest_int(name="max_leaf_nodes", low=2, high=100, step=2),
        'min_impurity_decrease': 0.013,
        'bootstrap': True,
        'random_state': SEED
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]  
    cv_scores = roc_auc_score(y_test, predictions)
    return np.mean(cv_scores)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)
# best_params = study.best_trial.params

# best_params = {'n_estimators': 37, 'max_depth' : 100, 'min_weight_fraction_leaf' : 0.0 , 'max_leaf_nodes': None, 'criterion': 'entropy', 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': True}
best_params = {'n_estimators': 37, 'criterion': 'entropy', 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': True}
print(best_params)
model = RandomForestClassifier(**best_params, random_state=SEED)
model.fit(X_train, y_train)
predictions = model.predict_proba(X_test)[:, 1]  
score = roc_auc_score(y_test, predictions)
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

submission = pd.DataFrame([best_params_ordered])
submission.to_csv(path + f'sample_submission_pred{score}.csv', index=False)

'''
{'n_estimators': 52, 'criterion': 'entropy', 'min_samples_split': 197, 'min_samples_leaf': 77, 'max_features': 'log2', 'min_impurity_decrease': 4.6398069476440685e-05, 'bootstrap': True} 

{'n_estimators': 52, 'criterion': 'gini', 'min_samples_split': 114, 'min_samples_leaf': 6, 'max_features': None, 'min_impurity_decrease': 0.00015374456272445724, 'bootstrap': True}
ROC AUC score :  0.8425974025974027  
'''


