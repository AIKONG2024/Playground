from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

#Modeling
def get_model(params, model):
    return model(**params)

def get_catboost(params):
    return get_model(params, CatBoostClassifier)

def get_xgboost(params):
    return get_model(params, XGBClassifier)

def get_lightgbm(params):
    return get_model(params, LGBMClassifier)

def get_randomForest(params):
    return get_model(params, RandomForestClassifier)

#Model&fit
def get_fitted_catboost(params,  X_train, X_test, y_train, y_test, features) -> CatBoostClassifier:
    clf = get_catboost(params)
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features= features, verbose=False)
    return clf

def get_fitted_xgboost(params,  X_train, X_test, y_train, y_test) -> XGBClassifier:
    clf = get_xgboost(params)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],verbose=False)
    return clf

def get_fitted_lightgbm(params, X_train, X_test, y_train, y_test) -> LGBMClassifier:
    clf = get_lightgbm(params)
    clf.fit(X_train, y_train, eval_set=(X_test, y_test))
    return clf

def get_fitted_randomForest(params,  X_train, X_test, y_train, y_test) -> RandomForestClassifier:
    clf = get_randomForest(params)
    clf.fit(X_train, y_train)
    return clf

