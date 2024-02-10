import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
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

def preprocessing(data):
    #BMI
    data['BMI'] =  data['Weight'] / (data['Height'] ** 2)
    
    #CALC, CAEC
    levels = {"Always": 3, "Frequently": 2, "Sometimes": 1, "no": 0}
    data["CALC"] = data["CALC"].map(levels)
    data["CAEC"] = data["CAEC"].map(levels)
    
    #Meal_Habits
    data['Meal_Habits'] = data['FCVC'] * data["NCP"]
    return data
    
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
    

def scaling(scaler, data):
    if scaler is None :
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        # scaler = MaxAbsScaler()
        # scaler = RobustScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    return data, scaler

def smote(x,y):
    return SMOTE(random_state=SEED).fit_resample(x,y)

def get_data(train_data):
    X, y = train_data.drop(["NObeyesdad"], axis=1), train_data.NObeyesdad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    return (X_train, X_test, y_train, y_test)
