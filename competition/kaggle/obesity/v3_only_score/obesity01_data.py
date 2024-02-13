import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from obesity00_constant import SEED


def lable_encoding(encoder, data):
    if encoder is None :
        encoder = LabelEncoder()
        data = encoder.fit_transform(data.astype(str))
    else:
        data = encoder.fit_transform(data.astype(str))
    return data, encoder

def x_preprocessing(data:pd.DataFrame):
    # dataFame['BMI'] =  dataFame['Weight'] / (dataFame['Height'] ** 2)
    levels = {"Always": 3, "Frequently": 2, "Sometimes": 1, "no": 0}
    data["CALC"] = data["CALC"].map(levels)
    data["CAEC"] = data["CAEC"].map(levels)
    # data['Meal_Habits'] = data['FCVC'] * data["NCP"]
    return data

def train_only_preprocessing(data:pd.DataFrame):
    data = data.drop_duplicates()
    data = data[data["Age"] < 46]
    return data  

def y_encoding(data:pd.DataFrame):
    label_dict = {
    'Insufficient_Weight': 0, 
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3, 
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5, 
    'Obesity_Type_III': 6
    }
    inverse_label_dict = {label_dict[x]:x for x in label_dict.keys()}
    data = data.map(label_dict)
    return data , inverse_label_dict

def get_data(data:pd.DataFrame):
    X, y = data.drop(["NObeyesdad"], axis=1), data.NObeyesdad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    return (X_train, X_test, y_train, y_test)

def scaling(scaler, data:pd.DataFrame):
    if scaler is None :
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        # scaler = MaxAbsScaler()
        # scaler = RobustScaler()
        data = scaler.fit_transform(data.values.reshape(-1,1))
    else:
        data = scaler.transform(data.values.reshape(-1,1))
    return data, scaler