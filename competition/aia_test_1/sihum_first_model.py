import pandas as pd
import numpy as np
import time
from sklearn.metrics import r2_score
import sys
sys.path.append('C:/MyPackages/')
from keras_custom_pk.hyper_model import MulticlassClassificationModel
from keras_custom_pk.file_name import *
from keras_custom_pk.callbacks import CustomEarlyStoppingAtLoss

time_steps = 180
behind_size = 2 
compare_predict_size = 10

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

'''
전처리 note 
===================================
삼성
 
2015/08/30 이후 데이터 사용
7일
30일

===================================
아모레 

2020/03/23 이후 데이터 사용
7일
30일

===================================
'''

# ===========================================================================
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
# 수치화 - 문자: 전일비
# from sklearn.preprocessing import LabelEncoder
# lbe = LabelEncoder()
# samsung_csv['전일비'] = lbe.fit_transform(samsung_csv['전일비'])
# amore_csv['전일비'] = lbe.fit_transform(amore_csv['전일비'])

# ============================================================================
# split
samsung_x, samsung_y = split_xy(samsung_csv, time_steps, behind_size, ['시가'])
amore_x, amore_y = split_xy(amore_csv, time_steps, behind_size, ['종가'])
# 샘플 추출 x
samsung_sample_x = samsung_x[-compare_predict_size:]
amore_sample_x = amore_x[-compare_predict_size :]
# 샘플 추출 y
samsung_sample_y = np.append(samsung_y[-compare_predict_size + 2:], ["24/02/06 시가","24/02/07 시가"]) 
amore_sample_y = np.append(amore_y[-compare_predict_size + 2:], ["24/02/06 종가","24/02/07 종가"]) 

# ============================================================================
# 데이터셋 나누기
from sklearn.model_selection import train_test_split
s_x_train, s_x_test, s_y_train, s_y_test = train_test_split(samsung_x,samsung_y, train_size=0.8, shuffle=True, random_state=1234)
a_x_train, a_x_test, a_y_train, a_y_test = train_test_split(amore_x,amore_y, train_size=0.8, shuffle=True, random_state=1234)

# ============================================================================
# 스케일링
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler , RobustScaler
r_s_x_train = s_x_train.reshape(s_x_train.shape[0],s_x_train.shape[1] * s_x_train.shape[2])
r_s_x_test = s_x_test.reshape(s_x_test.shape[0], s_x_test.shape[1] * s_x_test.shape[2])
r_a_x_train = a_x_train.reshape(a_x_train.shape[0], a_x_train.shape[1] * a_x_train.shape[2])
r_a_x_test = a_x_test.reshape(a_x_test.shape[0], a_x_test.shape[1] * a_x_test.shape[2])
r_samsung_sample_x = samsung_sample_x.reshape(samsung_sample_x.shape[0], samsung_sample_x.shape[1] * samsung_sample_x.shape[2])
r_amore_sample_x = amore_sample_x.reshape(amore_sample_x.shape[0], amore_sample_x.shape[1] * amore_sample_x.shape[2])

samsung_scaler = StandardScaler()
r_s_x_train = samsung_scaler.fit_transform(r_s_x_train)
r_s_x_test = samsung_scaler.transform(r_s_x_test)
r_samsung_sample_x = samsung_scaler.transform(r_samsung_sample_x)
amore_scaler = StandardScaler()
r_a_x_train = amore_scaler.fit_transform(r_a_x_train)
r_a_x_test = amore_scaler.transform(r_a_x_test)
r_amore_sample_x = amore_scaler.transform(r_amore_sample_x)

# ============================================================================
# reshape
#3차원
s_x_train = r_s_x_train.reshape(-1, s_x_train.shape[1], s_x_train.shape[2])
s_x_test = r_s_x_test.reshape(-1, s_x_test.shape[1], s_x_test.shape[2])
a_x_train = r_a_x_train.reshape(-1, a_x_train.shape[1], a_x_train.shape[2])
a_x_test = r_a_x_test.reshape(-1, a_x_test.shape[1], a_x_test.shape[2])
samsung_sample_x = r_samsung_sample_x.reshape(-1, samsung_sample_x.shape[1], samsung_sample_x.shape[2])
amore_sample_x = r_amore_sample_x.reshape(-1, amore_sample_x.shape[1], amore_sample_x.shape[2])

#4차원
# s_x_train = r_s_x_train.reshape(-1, s_x_train.shape[1], s_x_train.shape[2], 1)
# s_x_test = r_s_x_test.reshape(-1, s_x_test.shape[1], s_x_test.shape[2], 1)
# a_x_train = r_a_x_train.reshape(-1, a_x_train.shape[1], a_x_train.shape[2], 1)
# a_x_test = r_a_x_test.reshape(-1, a_x_test.shape[1], a_x_test.shape[2], 1)
# samsung_sample_x = r_samsung_sample_x.reshape(-1, samsung_sample_x.shape[1], samsung_sample_x.shape[2], 1)
# amore_sample_x = r_amore_sample_x.reshape(-1, amore_sample_x.shape[1], amore_sample_x.shape[2], 1)

# ============================================================================
# 2. 모델 구성
from keras.models import Model
from keras.layers import Dense, LSTM, ConvLSTM1D, Input, Flatten, concatenate, MaxPooling1D
from keras.callbacks import EarlyStopping

# ============================================================================
#모델 삼성
s_input = Input(shape=(time_steps, len(samsung_csv.columns)))
s_layer_1 = LSTM(32)(s_input)
s_layer_2 = Dense(16,activation='relu')(s_layer_1)
s_layer_3 = Dense(16,activation='relu')(s_layer_2)
s_layer_4 = Dense(16,activation='relu')(s_layer_3)
s_layer_5 = Dense(16,activation='relu')(s_layer_4)
s_output = Dense(16)(s_layer_5)

# s_input = Input(shape=(time_steps, len(samsung_csv.columns), 1))
# s_layer_1 =  ConvLSTM1D(filters=32, kernel_size=2)(s_input)
# s_mp_layer = MaxPooling1D(2)(s_layer_1)
# s_flatter = Flatten()(s_mp_layer)
# s_layer_2 = Dense(8,activation='relu')(s_flatter)
# s_layer_3 = Dense(16,activation='relu')(s_layer_2)
# s_layer_4 = Dense(8,activation='relu')(s_layer_3)
# s_layer_5 = Dense(16,activation='relu')(s_layer_4)
# s_layer_6 = Dense(8,activation='relu')(s_layer_5)
# s_layer_7 = Dense(16, activation='relu')(s_layer_6)
# s_layer_8 = Dense(8, activation="relu")(s_layer_7)
# s_output = Dense(16)(s_layer_8)

# ============================================================================
#모델 아모레퍼시픽
a_input = Input(shape=(time_steps, len(amore_csv.columns)))
a_layer_1 = LSTM(32)(a_input)
a_layer_2 = Dense(16, activation='relu')(a_layer_1)
a_layer_3 = Dense(16, activation='relu')(a_layer_2)
a_layer_4 = Dense(16, activation='relu')(a_layer_3)
a_layer_5 = Dense(16, activation='relu')(a_layer_4)
a_output = Dense(16)(a_layer_5)


# a_input = Input(shape=(time_steps, len(amore_csv.columns), 1))
# a_layer_1 =  ConvLSTM1D(filters=32, kernel_size=3)(s_input)
# a_mp_layer = MaxPooling1D(2)(a_layer_1)
# a_flatter = Flatten()(a_mp_layer)
# a_layer_2 = Dense(16, activation='relu')(a_flatter)
# a_layer_3 = Dense(8, activation='relu')(a_layer_2)
# a_layer_4 = Dense(16, activation='relu')(a_layer_3)
# a_layer_5 = Dense(8, activation='relu')(a_layer_4)
# a_layer_6 = Dense(16, activation='relu')(a_layer_5)
# a_layer_7 = Dense(8, activation='relu')(a_layer_6)
# a_layer_8 = Dense(16, activation='relu')(a_layer_7)
# a_layer_9 = Dense(8, activation='relu')(a_layer_8)
# a_layer_10 = Dense(16, activation='relu')(a_layer_9)
# a_layer_11 = Dense(8, activation='relu')(a_layer_10)
# a_output = Dense(16)(a_layer_11)

# ============================================================================
#merge 1
m1_layer_1 = concatenate([s_output, a_output])
m1_layer_2 = Dense(64, activation='relu')(m1_layer_1)
m1_layer_3 = Dense(32 ,activation='relu')(m1_layer_2)
m1_last_output = Dense(1)(m1_layer_3)
m2_last_output = Dense(1)(m1_layer_3)


# m1_layer_2 = Dense(128, activation='relu')(m1_layer_1)
# m1_layer_3 = Dense(16 ,activation='relu')(m1_layer_2)
# m1_layer_4 = Dense(16 ,activation='relu')(m1_layer_3)
# m1_layer_5 = Dense(16)(m1_layer_4)
# m1_layer_6 = Dense(16)(m1_layer_5)
# m1_layer_7 = Dense(16)(m1_layer_6)
# m1_layer_8 = Dense(16)(m1_layer_7)
# m1_layer_9 = Dense(16)(m1_layer_8)
# m1_last_output = Dense(1)(m1_layer_9)
# m2_last_output = Dense(1)(m1_layer_9)

# # ============================================================================
# #merge 2
# m2_layer_1 = concatenate([s_output, a_output]) 
# m2_layer_2 = Dense(16)(m2_layer_1)
# m2_layer_3 = Dense(16)(m2_layer_2)
# m2_layer_4 = Dense(16)(m2_layer_3)
# m2_layer_5 = Dense(16)(m2_layer_4)
# m2_layer_6 = Dense(16)(m2_layer_5)
# m2_layer_7 = Dense(16)(m2_layer_6)
# m2_layer_8 = Dense(16)(m2_layer_7)
# m2_last_output = Dense(1)(m2_layer_5)

# ============================================================================
#model 정의
model = Model(inputs = [s_input, a_input], outputs = [m1_last_output, m2_last_output])

# ============================================================================
# 3. 컴파일 훈련
model.compile(loss='mae', optimizer='adam')
while 1 : 
    model.fit(
            [s_x_train, a_x_train],
            [s_y_train, a_y_train],
            epochs=1000,
            batch_size=3000,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                CustomEarlyStoppingAtLoss(
                    patience=2000,
                    monitor="val_loss",
                    overfitting_stop_line=0,
                    overfitting_count=1000,
                    stop_tranning_epoch=100,
                    stop_tranning_value=30000,
                    is_log=True,
                )
            ],
        )

    # 4. 평가 예측
    # ============================================================================
    # evaluate 평가, r2 스코어
    loss = model.evaluate([s_x_test, a_x_test], [s_y_test, a_y_test])
    predict = model.predict([s_x_test, a_x_test])
    s_r2 = r2_score(s_y_test, predict[0])
    a_r2 = r2_score(a_y_test, predict[1])
    print("="*100)
    print(f"합계 loss : {loss[0]} / 삼성 loss : {loss[1]} / 아모레 loss : {loss[2]}" )
    print(f"삼성 r2 : {s_r2} / 아모레 r2 : {a_r2}")
    print("="*100)

    # ============================================================================
    # 최근 실제 값과 비교 (compare_predict_size = ?)
    sample_dataset_y = [samsung_sample_y,amore_sample_y]
    sample_predict_x = model.predict([samsung_sample_x, amore_sample_x])

    print("="*100)
    for i in range(len(sample_dataset_y)):
        if i == 0 :
            print("\t\tSAMSUNG\t시가")
        else:
            print("="*100)
            print("\t\tAMORE\t종가")
        for j in range(compare_predict_size):
            print(f"\tD-{compare_predict_size - j  - 1}: {sample_dataset_y[i][j]}\t예측값 {np.round(sample_predict_x[i][j],2)}\t")
    print("="*100)

    # ============================================================================
    # .h5 file 저장
    if 73400 < sample_predict_x[0][compare_predict_size-1] <74200 and 123500 < sample_predict_x[1][compare_predict_size-1] <128000  :
        h_path = "C:/_data/sihum/save_weight/"
        h5_file_name_d = h5_file_name(h_path , f"save_model_samsung_{sample_predict_x[0][len(sample_predict_x)-1]}_amore_{sample_predict_x[1][len(sample_predict_x)-1]}_")
        model.save(h5_file_name_d)
        print("="*100)
        print("\t\t.h5파일저장완료")
        break
