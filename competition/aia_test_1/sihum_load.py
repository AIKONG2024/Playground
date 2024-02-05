import pandas as pd
import numpy as np
import time
import sys

time_steps = 5
behind_size = 2 

def split_xy(dataFrame, cutting_size, y_behind_size,  y_column):
    split_start_time = time.time()
    xs = []
    ys = [] 
    for i in range(len(dataFrame) - cutting_size - y_behind_size):
        x = dataFrame[i : (i + cutting_size)]
        y = dataFrame[i + cutting_size + y_behind_size : (i + cutting_size + y_behind_size + 1) ][y_column]
        xs.append(x)
        ys.append(y)
    split_end_time = time.time()
    print("spliting time : ", np.round(split_end_time - split_start_time, 2),  "sec")
    return (np.array(xs), np.array(ys).reshape(-1,1))

# ===========================================================================
# 데이터 저장
path = "C:/_data/sihum/"
samsung_csv = pd.read_csv(path + "삼성 240205.csv", encoding='cp949', thousands=',', index_col=0)
amore_csv = pd.read_csv(path + "아모레 240205.csv", encoding='cp949', thousands=',', index_col=0)

# ===========================================================================
# 데이터 일자 이후 자르기
samsung_csv = samsung_csv[samsung_csv.index > "2015/08/29"]
amore_csv = amore_csv[amore_csv.index > "2015/08/29"]

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

# ============================================================================
# 수치화 - 문자: 전일비
from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
samsung_csv['전일비'] = lbe.fit_transform(samsung_csv['전일비'])
amore_csv['전일비'] = lbe.fit_transform(amore_csv['전일비'])
# ============================================================================
# split
samsung_x, samsung_y = split_xy(samsung_csv, time_steps, behind_size, ['시가'])
amore_x, amore_y = split_xy(amore_csv, time_steps, behind_size, ['종가'])

# 샘플 추출
samsung_sample_x = samsung_x[-5:]
samsung_sample_y = samsung_y[-5:]
amore_sample_x = amore_x[-5:]
amore_sample_y = amore_y[-5:]
# ============================================================================
# 데이터셋 나누기
from sklearn.model_selection import train_test_split
s_x_train, s_x_test, s_y_train, s_y_test = train_test_split(samsung_x,samsung_y, train_size=0.8, shuffle=False, random_state=1234)
a_x_train, a_x_test, a_y_train, a_y_test = train_test_split(amore_x,amore_y, train_size=0.8, shuffle=False, random_state=1234)

# ============================================================================
# 스케일링
from sklearn.preprocessing import StandardScaler, MinMaxScaler
r_s_x_train = s_x_train.reshape(s_x_train.shape[0],s_x_train.shape[1] * s_x_train.shape[2])
r_s_x_test = s_x_test.reshape(s_x_test.shape[0], s_x_test.shape[1] * s_x_test.shape[2])
r_a_x_train = a_x_train.reshape(a_x_train.shape[0], a_x_train.shape[1] * a_x_train.shape[2])
r_a_x_test = a_x_test.reshape(a_x_test.shape[0], a_x_test.shape[1] * a_x_test.shape[2])
r_samsung_sample_x = samsung_sample_x.reshape(samsung_sample_x.shape[0], samsung_sample_x.shape[1] * samsung_sample_x.shape[2])
r_amore_sample_x = amore_sample_x.reshape(amore_sample_x.shape[0], amore_sample_x.shape[1] * amore_sample_x.shape[2])

samsung_scaler = MinMaxScaler()
r_s_x_train = samsung_scaler.fit_transform(r_s_x_train)
r_s_x_test = samsung_scaler.transform(r_s_x_test)
r_samsung_sample_x = samsung_scaler.transform(r_samsung_sample_x)
amore_scaler = MinMaxScaler()
r_a_x_train = amore_scaler.fit_transform(r_a_x_train)
r_a_x_test = amore_scaler.transform(r_a_x_test)
r_amore_sample_x = amore_scaler.transform(r_amore_sample_x)
s_x_train = r_s_x_train.reshape(-1, s_x_train.shape[1], s_x_train.shape[2], 1)
s_x_test = r_s_x_test.reshape(-1, s_x_test.shape[1], s_x_test.shape[2], 1)
a_x_train = r_a_x_train.reshape(-1, a_x_train.shape[1], a_x_train.shape[2], 1)
a_x_test = r_a_x_test.reshape(-1, a_x_test.shape[1], a_x_test.shape[2], 1)

samsung_sample_x = r_samsung_sample_x.reshape(-1, samsung_sample_x.shape[1], samsung_sample_x.shape[2], 1)
amore_sample_x = r_amore_sample_x.reshape(-1, amore_sample_x.shape[1], amore_sample_x.shape[2], 1)

# 모델 구성
from keras.models import load_model

h_path = "C:/_data/sihum/save_weight/"
model = load_model(h_path + "weight_samsung.h5")

# 4. 평가 예측
loss = model.evaluate([s_x_test, a_x_test], [s_y_test, a_y_test])
print("loss :", loss)
predict = model.predict([s_x_test, a_x_test])
print(np.array(s_y_test).shape)

from sklearn.metrics import r2_score
s_r2 = r2_score(s_y_test, predict[0])
a_r2 = r2_score(a_y_test, predict[1])
print(f"삼성 r2 : {s_r2} / 아모레 r2 : {a_r2}")

# ============================================================================
print(f"삼성전자 2월 7일 시가 [{}]")
print("=====================================================================")

# loss : [645219.5625, 565802.375, 79417.1953125]
