import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from obesity00_constant import SEED


def lable_encoding(encoder, data):
    if encoder is None :
        encoder = LabelEncoder()
        data = encoder.fit_transform(data.astype(str))
    else:
        data = encoder.fit_transform(data.astype(str))
    return data, encoder

def x_preprocessing(dataFame):
    dataFame = dataFame[dataFame["Age"] < 46]
    # train_csv['BMI'] =  train_csv['Weight'] / (train_csv['Height'] ** 2)
    # test_csv['BMI'] =  test_csv['Weight'] / (test_csv['Height'] ** 2)
    # levels = {"Always": 3, "Frequently": 2, "Sometimes": 1, "no": 0}
    # dataFame["CALC"] = dataFame["CALC"].map(levels)
    # dataFame["CAEC"] = dataFame["CAEC"].map(levels)
    # dataFame["CALC"] = dataFame["CALC"].map(levels)
    # dataFame["CAEC"] = dataFame["CAEC"].map(levels)
    
    #Meal_Habits
    # train_csv['Meal_Habits'] = train_csv['FCVC'] * train_csv["NCP"]
    # test_csv['Meal_Habits'] = test_csv['FCVC'] * test_csv["NCP"]\
    return dataFame

def y_encoding(data):
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

def get_data(train_data):
    X, y = train_data.drop(["NObeyesdad"], axis=1), train_data.NObeyesdad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    return (X_train, X_test, y_train, y_test)
