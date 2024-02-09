from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from obesity00_seed import SEED

#Modeling
def get_catboost(params , patience, iterations):
    clf = CatBoostClassifier(
        **params,
        task_type="GPU",
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        early_stopping_rounds=patience,
        iterations=iterations,
        random_state=SEED,
    )
    return clf

def get_xgboost(params , patience):
    clf = XGBClassifier(
        **params,
        tree_method="gpu_hist",
        objective = "multi:sotfmax",
        device = 'cuda',
        enable_categorical=True,
        max_cat_to_onehot=7,
        early_stopping_rounds=patience,
        importance_type = 'weight',
        random_state=SEED,
    )
    return clf

def get_lightgbm(params , patience, iterations):
    clf = LGBMClassifier(
        **params,
        task_type="GPU",
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        early_stopping_rounds=patience,
        iterations=iterations,
        random_state=SEED,
    )
    return clf

#Model&fit
def get_fitted_catboost(params, datasets, patience, iterations, features) -> CatBoostClassifier:
    X_train, X_test, y_train, y_test = datasets
    clf = get_catboost(params, patience, iterations)
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features= features, verbose=False)
    return clf

def get_fitted_xgboost(params, datasets, patience) -> XGBClassifier:
    X_train, X_test, y_train, y_test = datasets
    clf = get_xgboost(params, patience)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)] ,verbose=False)
    return clf

def get_fitted_lightgbm(params, datasets, patience, iterations, features) -> LGBMClassifier:
    X_train, X_test, y_train, y_test = datasets
    clf = get_lightgbm(params, patience, iterations)
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features= features, verbose=False)
    return clf

# test
# test = lambda func: func()
# from optuna import Trial
# test(get_fitted_catboost(Trial.suggest_float("learning_rate", 1e-3, 1e-1)))