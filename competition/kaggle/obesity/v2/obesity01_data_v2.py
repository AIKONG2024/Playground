import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from obesity00_seed_v2 import SEED


def lable_encoding(encoder, data):
    if encoder is None :
        encoder = LabelEncoder()
        data = encoder.fit_transform(data.astype(str))
    else:
        data = encoder.fit_transform(data.astype(str))
    return data, encoder

def onehot_encoding(encoder, data) :
    if encoder is None :
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        data = encoder.fit_transform(data.astype(str).values.reshape(-1,1))
    else:
        data = encoder.transform(data.astype(str).values.reshape(-1,1))
    return data, encoder

def scaling(scaler, data):
    if scaler is None :
        # scaler = MinMaxScaler()
        # scaler = StandardScaler()
        scaler = MaxAbsScaler()
        # scaler = RobustScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    return data, scaler

def get_data(train_data):
    X, y = train_data.drop(["NObeyesdad"], axis=1), train_data.NObeyesdad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    return (X_train, X_test, y_train, y_test)
