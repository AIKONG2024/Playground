from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Modeling
def get_catboost(params):
    clf = CatBoostClassifier(
        **params,
    )
    return clf

def get_xgboost(params):
    clf = XGBClassifier(
        **params
    )
    return clf

def get_lightgbm(params):
    clf = LGBMClassifier(
        **params,
    )
    return clf

def get_randomForest(params):
    clf = RandomForestClassifier(
        **params,
    )
    return clf

def get_logisticRegressor(params):
    clf = LogisticRegression(
        **params,
    )
    return clf

#Model&fit
def get_fitted_catboost(params, datasets, features) -> CatBoostClassifier:
    X_train, X_test, y_train, y_test = datasets
    clf = get_catboost(params)
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features= features, verbose=False)
    return clf

def get_fitted_xgboost(params, datasets) -> XGBClassifier:
    X_train, X_test, y_train, y_test = datasets
    clf = get_xgboost(params)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)] ,verbose=False)
    return clf

def get_fitted_lightgbm(params, datasets) -> LGBMClassifier:
    X_train, X_test, y_train, y_test = datasets
    clf = get_lightgbm(params)
    clf.fit(X_train, y_train, eval_set=(X_test, y_test))
    return clf

def get_fitted_randomForest(params, datasets) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = datasets
    clf = get_randomForest(params)
    clf.fit(X_train, y_train)
    return clf


# test
# test = lambda func: func()
# from optuna import Trial
# test(get_fitted_catboost(Trial.suggest_float("learning_rate", 1e-3, 1e-1)))