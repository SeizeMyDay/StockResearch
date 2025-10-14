2025.09.08 ~ 09.22  
# SR Warm-up Study: Jane Street Real-Time Market Data Forecasting
Overview: <https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting>  
작업물: <https://www.kaggle.com/code/injchoi/elasticnetcv>

로직  

1. 파티션 구분 없이 다 합쳐진 데이터 symbol별로 다시 나눔. symbol_0부터 symbol_38까지 총 39개 심볼 존재.

2. 심볼별로 ElasticNet을 통해 day, feature당 response_6과의 상관정도(coefficient)추정. feature는 총 79개 존재. 심볼은 symbol_0의 경우 1300여 개 day 존재. 심볼별로 79 × 1300(+α) 데이터셋 39개 생성.

3. feature별로 coefficient 집계. 컬럼이 symbol, 인덱스가 feature인 데이터프레임 1개로 만듦.

4. 심볼별로 상위 30개 feature 선정, 심볼별로 따로 예측 수행.
