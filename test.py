import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 예제 데이터를 생성합니다.
dates = pd.date_range(start='2020-01-01', periods=100)
np.random.seed(0)
prices = np.random.normal(loc=100, scale=1, size=len(dates)).cumsum()
data = pd.DataFrame(data={'Close': prices}, index=dates)

# MACD 계산 함수
def calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
    data['EMA_Fast'] = data['Close'].ewm(span=fastperiod, adjust=False).mean()
    data['EMA_Slow'] = data['Close'].ewm(span=slowperiod, adjust=False).mean()
    data['MACD'] = data['EMA_Fast'] - data['EMA_Slow']
    data['Signal'] = data['MACD'].ewm(span=signalperiod, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal']
    return data

# MACD 계산
data = calculate_macd(data)

# 다이버전스 확인 함수
def find_divergence(data, macd_col='MACD', price_col='Close'):
    divergences = []
    for i in range(1, len(data) - 1):
        # 가격이 낮아지고 있는 경우 (하락 다이버전스)
        if data[price_col].iloc[i] < data[price_col].iloc[i - 1] and data[macd_col].iloc[i] > data[macd_col].iloc[i - 1]:
            divergences.append((data.index[i], 'Bullish Divergence'))
        # 가격이 높아지고 있는 경우 (상승 다이버전스)
        elif data[price_col].iloc[i] > data[price_col].iloc[i - 1] and data[macd_col].iloc[i] < data[macd_col].iloc[i - 1]:
            divergences.append((data.index[i], 'Bearish Divergence'))
    return divergences

# 다이버전스 찾기
divergences = find_divergence(data)

# 다이버전스 출력
for divergence in divergences:
    print(f"Date: {divergence[0]}, Type: {divergence[1]}")

# 시각화
plt.figure(figsize=(14, 7))

# 가격 차트
plt.subplot(2, 1, 1)
plt.plot(data['Close'], label='Close Price')
plt.title('Price Chart')
plt.legend()

# MACD 차트
plt.subplot(2, 1, 2)
plt.plot(data['MACD'], label='MACD')
plt.plot(data['Signal'], label='Signal')
plt.bar(data.index, data['MACD_Hist'], label='MACD Histogram')
for divergence in divergences:
    plt.axvline(x=divergence[0], color='red' if divergence[1] == 'Bearish Divergence' else 'green', linestyle='--')
plt.title('MACD Chart')
plt.legend()

plt.tight_layout()
plt.show()