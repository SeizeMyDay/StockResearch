#딥러닝 주가예측
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt
from Investar import Analyzer
from tensorflow.keras import losses

mk = Analyzer.MarketDB()
raw_df = mk.get_daily_price('000670', '2020-05-03', '2022-05-03')

window_size = 5 #이전 15일 데이터를 가지고 예측**
data_size = 5


# 활성화함수
# actf_1: 기본 활성화함수, actf_2: 순환 활성화함수
(actf_1, actf_2) = ('tanh', 'hard_sigmoid')
# 1                ('selu', 'selu')
# 2                ('tanh', 'selu')
# 3                ('softsign', 'selu')
# 4                ('elu', 'elu')
# 5                ('elu', 'relu')
# 6                ('tanh', 'softplus')
# 7                ('tanh', 'relu')
# 8                ('selu', 'softsign')
'''
actf_set =
actf_list = [('selu', 'selu'), ('tanh', 'selu'), ('softsign', 'selu'), ('elu', 'elu'), ('elu', 'relu'), ('tanh', 'softplus'), ('tanh', 'relu'), ('selu', 'softsign')]

for i in range(len(actf_list)) :
    if actf_set == i+1:
        (actf_1, actf_2) = actf_list[i]
'''

#최적화함수
optf = 'Adam'
# 1    'RMSprop'
# 2    'Adagrad'
# 3    'Adadelta'
# 4    'Adam'
# 5    'Adamax'
# 6    'Nadam'


def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

dfx = raw_df[['open','high','low','volume', 'close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['close']]

x = dfx.values.tolist()
y = dfy.values.tolist()#y는 x를 scaling한것

data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] # 미만이니까, 다음 날 종가는 포함x
    _y = y[i + window_size]     # 다음 날 종가 - 이것 5일으로 변경?
    data_x.append(_x)
    data_y.append(_y)
#print(_x, "->", _y)#매핑 확인 : 매핑을 바꿔서 5일 예측? 안되면 일요일에 예측- 월요일 확인 이런식으로 논문 완성
#과적합 현상 방지를 위해, 70%는 훈련데이터. 30%는 테스트 데이터** - 이것도 변경 생각
train_size = int(len(data_y) * 0.7)
train_x = np.array(data_x[0 : train_size])#이부분을  변경할 생각 - 3분할?
train_y = np.array(data_y[0 : train_size])

test_size = len(data_y) - train_size
test_x = np.array(data_x[train_size : len(data_x)])
test_y = np.array(data_y[train_size : len(data_y)])

# 모델 생성 - 활성화함수 sigmoid 포함 다른것들을 적용 가능*
model = Sequential()
model.add(LSTM(units=5, activation=actf_1, return_sequences=True, input_shape=(window_size, data_size)))
model.add(Dropout(0.1))#10% 드랍해서 과적합 방지
model.add(LSTM(units=5, activation=actf_2))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer=optf, loss=losses.mean_squared_error)
model.fit(train_x, train_y, epochs=100, batch_size=32)
pred_y = model.predict(test_x)

print("activation function: (%s, %s)" %(actf_1, actf_2))
print("optimizer: %s" %optf)


#결과 plot
plt.figure()
plt.plot(test_y, color='red', label='real stock price')
plt.plot(pred_y, color='blue', label='predicted stock price')
plt.title('stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()

# raw_df.close[-1] : dfy.close[-1] = x : pred_y[-1]
print("Tomorrow's price :", raw_df.close[-1] * pred_y[-1] / dfy.close[-1], 'KRW')
