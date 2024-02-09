import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def encoding(train_data, test_data):
    encoder = LabelEncoder()
    train_data["Gender"] = encoder.fit_transform(train_data["Gender"])
    test_data["Gender"] = encoder.transform(test_data["Gender"])

    train_data["family_history_with_overweight"] = encoder.fit_transform(train_data["family_history_with_overweight"])
    test_data["family_history_with_overweight"] = encoder.transform(test_data["family_history_with_overweight"])

    train_data["FAVC"] = encoder.fit_transform(train_data["FAVC"])
    test_data["FAVC"] = encoder.transform(test_data["FAVC"])

    train_data["CAEC"] = encoder.fit_transform(train_data["CAEC"])
    test_data["CAEC"] = encoder.transform(test_data["CAEC"])

    train_data["SMOKE"] = encoder.fit_transform(train_data["SMOKE"])
    test_data["SMOKE"] = encoder.transform(test_data["SMOKE"])

    train_data["SCC"] = encoder.fit_transform(train_data["SCC"])
    test_data["SCC"] = encoder.transform(test_data["SCC"])

    train_data["CALC"] = encoder.fit_transform(train_data["CALC"])
    if "Always" not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, "Always")
    test_data["CALC"] = encoder.transform(test_data["CALC"])

    train_data["MTRANS"] = encoder.fit_transform(train_data["MTRANS"])
    test_data["MTRANS"] = encoder.transform(test_data["MTRANS"])

    train_data["NObeyesdad"] = encoder.fit_transform(train_data["NObeyesdad"])
    return train_data, test_data, encoder


def get_data(train_data):
    from obesity03_train import SEED

    X, y = train_data.drop(["NObeyesdad"], axis=1), train_data.NObeyesdad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    return (X_train, X_test, y_train, y_test)
