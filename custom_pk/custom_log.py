#-v1 2024.01.15 KONG SEONUI
import pandas as pd
from custom_pk.custom_file_name import csv_file_name

def logging(path, log_numbers, ** parameters):
    for i in range(0, len(parameters.keys)):
        filename = csv_file_name(path, f'log_{log_numbers}')
        df_new = pd.DataFrame({}) 
    df_new.to_csv(path + f"{filename}.csv", mode= 'a', header=True)