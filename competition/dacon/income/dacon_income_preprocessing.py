#https://dacon.io/competitions/official/236230/overview/description

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def outliers(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return (data > upper_bound) | (data < lower_bound)

#데이터
path = 'C:/_data/dacon/income/'
SEED = 42

#read 
train_df = pd.read_csv(path + "train.csv", index_col=0)
test_df = pd.read_csv(path + "test.csv", index_col=0)

#확인
#이상치 처리
# gains_outlier_indexes = outliers(train_df['Gains'])
# train_df = train_df.drop(train_df.index[gains_outlier_indexes])
train_df = train_df[train_df['Losses'] < 3000]

#Age 분포도
# plt.figure(figsize=(14,10))
# plt.subplot(2,2,1)
# plt.hist(train_df['Age'], bins= 30, color = 'skyblue', edgecolor = 'black')
# plt.title('Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')

# # Gain 분포도
# plt.subplot(2,2,2)
# plt.hist(train_df['Gains'], bins = 30, color='skyblue', edgecolor = 'black')
# plt.title('Gains')
# plt.xlabel('Gains')
# plt.ylabel('Frequency')

# # Losses 분포도
# plt.subplot(2,2,3)
# plt.hist(train_df['Losses'], bins=30, color='skyblue', edgecolor = 'black')
# plt.title('Losses')
# plt.xlabel('Losses')
# plt.ylabel('Frequency')

# # Income 분포도
# plt.subplot(2,2,4)
# plt.hist(train_df['Income'], bins=30, color='skyblue', edgecolor = 'black')
# plt.title('Income')
# plt.xlabel('Income')
# plt.ylabel('Frequency')
# plt.show()

# plt.figure(figsize=(14,10))
# plt.subplot(2,1,1)
# plt.hist(train_df['Gains'], bins = 30, color='skyblue', edgecolor = 'black')
# plt.title('Train')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.subplot(2,1,2)
# plt.hist(test_df['Gains'], bins = 30, color='skyblue', edgecolor = 'black')
# plt.title('Test')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

#!!!
#Martial_Status - 순서형
print(np.unique(train_df['Citizenship']))

#인코딩
# 명목형 데이터 처리
nominal_columns = ['Gender', 'Race', 'Employment_Status', 'Industry_Status','Occupation_Status','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)']
one_hot_encoder = OneHotEncoder(sparse=False)
ohe_train = one_hot_encoder.fit_transform(train_df[nominal_columns])
ohe_test = one_hot_encoder.transform(test_df[nominal_columns])
nominal_train_df = pd.DataFrame(ohe_train, columns=one_hot_encoder.get_feature_names_out(nominal_columns), index=train_df.index)
nominal_test_df = pd.DataFrame(ohe_test, columns=one_hot_encoder.get_feature_names_out(nominal_columns), index=test_df.index)


# 순서형 데이터 처리
ordinal_columns = ['Education_Status', 'Income_Status', 'Martial_Status', 'Citizenship']
education_order = [
     'Children', 'Kindergarten', 'Elementary (1-4)', 'Elementary (5-6)', 'Middle (7-8)', 
    'High Freshman', 'High Sophomore', 'High Junior', 'High Senior', 'High graduate','College', 
    'Associates degree (Academic)', 'Associates degree (Vocational)',
    'Bachelors degree', 'Masters degree', 'Professional degree', 'Doctorate degree'
]
income_order = ['Unknown', 'Over Median', 'Under Median', 'Citizenship']
martial_staus_order = ['Single', 'Separated','Divorced', 'Widowed', 'Married (Armed Force Spouse)', 'Married','Married (Spouse Absent)']
citizen_order = ['Foreign-born (Naturalized US Citizen)', 'Foreign-born (Non-US Citizen)', 'Native (Born in Puerto Rico or US Outlying)','Native (Born Abroad)', 'Native']
oe = OrdinalEncoder(categories=[education_order, income_order, martial_staus_order, citizen_order])
oe_train = oe.fit_transform(train_df[ordinal_columns])
oe_test = oe.transform(test_df[ordinal_columns])
ordinal_train_df = pd.DataFrame(oe_train, columns=ordinal_columns, index=train_df.index)
ordinal_test_df = pd.DataFrame(oe_test, columns=ordinal_columns, index=test_df.index)
train_df = pd.concat([train_df.drop(columns=nominal_columns+ordinal_columns), nominal_train_df, ordinal_train_df], axis=1)
test_df = pd.concat([test_df.drop(columns=nominal_columns+ordinal_columns), nominal_test_df, ordinal_test_df], axis=1)

#그 외 범주형 데이터
lbe = LabelEncoder()
categorical_features = [col for col in train_df.columns if train_df[col].dtype == 'object']
for feature in categorical_features:
    lbe.fit(train_df[feature])
    train_df[feature] = lbe.transform(train_df[feature])
    new_labels = set(test_df[feature]) - set(lbe.classes_)
    for new_label in new_labels:
        lbe.classes_ = np.append(lbe.classes_, new_label)
    test_df[feature] = lbe.transform(test_df[feature])

#외의 데이터

#이상치 분석


#데이터 증폭

#추가 컬럼 생성

X = train_df.drop(['Income'], axis=1)
y = train_df['Income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

#스케일링
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)
x_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)

model = RandomForestRegressor(random_state=SEED)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f'rmse : {np.sqrt(mean_squared_error(pred, y_test))}')


#627
#625
#