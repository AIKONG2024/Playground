import pandas as pd
import pickle

import sys
sys.path.append("C:/MyPackages/")
from keras_custom_pk.file_name import csv_file_name, file_name
def save_submit(path, name, predictions):
    # ====================================================
    # 데이터 저장
    submission_csv = pd.read_csv(path + "sample_submission.csv")
    submission_csv["NObeyesdad"] = predictions
    file_name = csv_file_name(path, f"obesity_submit_{name}_")
    submission_csv.to_csv(file_name, index=False)
    print(
    f"""
    =============================================
    {file_name} 파일 저장 완료
    =============================================
    """
    )
    
def save_model(path, name, model):
    file_name_t = file_name(path, f"obesity_{name}_save_model", "model")
    pickle.dump(model, open(file_name_t, 'wb'))
    
def save_csv(path, name, csv):
    # ====================================================
    # 데이터 저장
    file_name = csv_file_name(path, f"obesity_submit_{name}_")
    csv.to_csv(file_name, index=False)
    print(
    f"""
    =============================================
    {file_name} 파일 저장 완료
    =============================================
    """
    )