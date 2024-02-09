from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

#Modeling
def get_catboost(params):
    from obesity03_train import SEED, PATIENCE, ITERATIONS
    clf = CatBoostClassifier(
        **params,
        task_type="GPU",
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        early_stopping_rounds=PATIENCE,
        iterations=ITERATIONS,
        random_state=SEED,
    )
    return clf

def get_xgboost(params):
    from obesity03_train import SEED, PATIENCE, ITERATIONS
    clf = XGBClassifier(
        **params,
        task_type="GPU",
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        early_stopping_rounds=PATIENCE,
        iterations=ITERATIONS,
        random_state=SEED,
    )
    return clf

def get_lightgbm(params):
    from obesity03_train import SEED, PATIENCE, ITERATIONS
    clf = LGBMClassifier(
        **params,
        task_type="GPU",
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        early_stopping_rounds=PATIENCE,
        iterations=ITERATIONS,
        random_state=SEED,
    )
    return clf

def get_fitted_catboost(params, datasets, cat_features) -> CatBoostClassifier:
    X_train, X_test, y_train, y_test = datasets
    clf = get_catboost(params)
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features= cat_features, verbose=False)
    return clf

def get_fitted_xgboost(params, datasets, cat_features) -> XGBClassifier:
    X_train, X_test, y_train, y_test = datasets
    clf = get_xgboost(params)
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features= cat_features, verbose=False)
    return clf

def get_fitted_lightgbm(params, datasets, cat_features) -> LGBMClassifier:
    X_train, X_test, y_train, y_test = datasets
    clf = get_lightgbm(params)
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features= cat_features, verbose=False)
    return clf

# test
# test = lambda func: func()
# from optuna import Trial
# test(get_fitted_catboost(Trial.suggest_float("learning_rate", 1e-3, 1e-1)))