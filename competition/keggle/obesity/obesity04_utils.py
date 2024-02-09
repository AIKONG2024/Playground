import pandas as pd
import sys
sys.path.append("C:/MyPackages/")
from keras_custom_pk.file_name import csv_file_name
def save(path, score, predictions):
    # ====================================================
    # 데이터 저장
    submission_csv = pd.read_csv(path + "sample_submission.csv")
    submission_csv["NObeyesdad"] = predictions
    file_name = csv_file_name(path, f"obesity_submit_{score}_")
    submission_csv.to_csv(file_name, index=False)
    print(
    f"""
    =============================================
    {file_name} 파일 저장 완료
    =============================================
    """
    )