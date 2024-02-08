import pandas as pd
import numpy as np
import time
from sklearn.metrics import r2_score
import sys
sys.path.append('C:/MyPackages/')
from keras_custom_pk.hyper_model import MulticlassClassificationModel
from keras_custom_pk.file_name import *
from keras_custom_pk.callbacks import CustomEarlyStoppingAtLoss
import random as rn
import tensorflow as tf
rn.seed(333)
tf.random.set_seed(123)
np.random.seed(321)

time_steps = 180
behind_size = 2 
compare_predict_size = 10

def split_xy(dataFrame, cutting_size, y_behind_size,  y_column):
    split_start_time = time.time()
    xs = []
    ys = [] 
    for i in range(len(dataFrame) - cutting_size - behind_size):
        x = dataFrame.iloc[i : i + cutting_size + behind_size + 2]
        y = dataFrame.iloc[i : i + cutting_size + behind_size]
        xs.append(x)
        ys.append(y)
    split_end_time = time.time()
    print("spliting time : ", np.round(split_end_time - split_start_time, 2),  "sec")
    return (np.array(xs), np.array(ys))


# 1. 데이터
path = "C:/_data/sihum/"
samsung_csv = pd.read_csv(path + "삼성 240205.csv", encoding='cp949', thousands=',', index_col=0)
amore_csv = pd.read_csv(path + "아모레 240205.csv", encoding='cp949', thousands=',', index_col=0)

# ===========================================================================
# 데이터 일자 이후 자르기
samsung_csv = samsung_csv[samsung_csv.index > "2018/08/30"][:956]
amore_csv = amore_csv[amore_csv.index > "2020/03/23"]

print(samsung_csv.columns)
# ['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#       '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],

print(samsung_csv.shape) #(2075, 16)
print(amore_csv.shape) #(2075, 16)

# ===========================================================================
# 결측치 처리
samsung_csv = samsung_csv.fillna(samsung_csv.ffill())
amore_csv = amore_csv.fillna(samsung_csv.ffill())

# ===========================================================================
# 데이터 역순 변환
samsung_csv.sort_values(['일자'], ascending=True, inplace=True)
amore_csv.sort_values(['일자'], ascending=True, inplace=True)

# ===========================================================================
# 컬럼 제거
samsung_csv = samsung_csv.drop('전일비', axis=1).drop('외인비', axis=1).drop('신용비', axis=1)
amore_csv = amore_csv.drop('전일비', axis=1).drop('외인비', axis=1).drop('신용비', axis=1)

# ============================================================================
# split
samsung_x, samsung_y = split_xy(samsung_csv, time_steps, behind_size, ['시가'])
amore_x, amore_y = split_xy(amore_csv, time_steps, behind_size, ['종가'])


print(samsung_x.shape)
print(samsung_y.shape)

print(samsung_x[:-1:][-1])
print(samsung_y[:-1][-1])
