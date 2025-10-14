#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  Install the custom Jane Street package (provided by competition organizers)
import os


# Kaggle's evaluation module for inference (do not modify)
from kaggle_evaluation import jane_street_inference_server


# Import essential libraries for data & ML
import numpy as np
import pandas as pd
import polars as pl       # optional, faster dataframe library
import torch              # for deep learning models
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV


# In[2]:


features = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/features.csv')
responders = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/responders.csv')
sample_submission = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/sample_submission.csv')

test_parquet = pl.read_parquet('/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet/date_id=0/part-0.parquet')

lags_parquet = pl.read_parquet('/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet/date_id=0/part-0.parquet')

# 전체 train_parquet, 이 중 partition_id = 6 까지만 사용할 것
train_parquet = []
for i in tqdm(range(10), desc='train_parquet') :
    file = f'/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet'
    train_parquet.append(pl.read_parquet(file))


# In[3]:


# 합치기
parquet = train_parquet[:7]
parquets = parquet[0]
for i in parquet[1:] :
    parquets = pl.concat([parquets, i])
parquets

# symbol_id별로 나누기
symbol_ids = parquets['symbol_id'].unique()

parquets_by_symbol_id = []

for i in tqdm(symbol_ids) :
    df = parquets.filter(pl.col('symbol_id') == i)
    parquets_by_symbol_id.append(df)


# In[4]:


# dump

#parquets_by_symbol_id
for i in tqdm(range(len(parquets_by_symbol_id))) :
    with open(f'parquets_by_symbol_id_{i}', 'wb') as f :
        pickle.dump(parquets_by_symbol_id[i], f)


# 1. 

# In[5]:


# ElasticNet 기반 coefficient의 절대값 반환 함수
# X: target_features_non_null.to_numpy()
# y: target.select(pl.col('responder_6')).to_numpy().ravel()
def EN(X, y) :
    model= ElasticNetCV(alphas=np.logspace(-4, 4, 50), l1_ratio=[.1, .5, .9], cv=5)
    model.fit(X, y)
    return pl.DataFrame(dict(zip(features_list, abs(model.coef_))))


# In[6]:


def coef(symbol_number) :
    # coefficient 저장용 데이터프레임 생성
    features_list = parquets_by_symbol_id[0].columns[4: -9]

    coefs_by_symbol = []

    dummy = {col:[0.0] for col in features_list}
    coefs = pl.DataFrame(dummy).clear()

    print(f'symbol {symbol_number}의 전체 일별 coefficient 도출')
    symbol = parquets_by_symbol_id[symbol_number]
    days = symbol['date_id'].unique().to_list()

    for day in tqdm(days, desc='일별 도출 중..') :
        target = symbol.filter(pl.col('date_id')==day)
        features_list = target.columns[4: -9]

        target_features = target.select(pl.col(features_list))

        target_features_non_null = target_features.select([
            pl.col(col).fill_null(pl.col(col).mean()) if target_features[col].null_count() < target_features.height
            else pl.col(col).fill_null(0)
            for col in target_features.columns
        ])

        target_features_non_null

        X = target_features_non_null.to_numpy()
        y = target.select(pl.col('responder_6')).to_numpy().ravel()

        coefs = coefs.vstack(EN(X, y))

    coefs_by_features = {}
    for col in coefs.columns :
        coefs_by_features[col] = coefs[col].sum()

    coefs_by_features_index = coefs.columns
    coefs_by_features_col = 'coef'
    coefs_by_features_row = []

    for col in coefs.columns :
        coefs_by_features_row.append(coefs[col].sum())

    coefs_by_features = pl.DataFrame({'index':coefs_by_features_index, 'coef': coefs_by_features_row})
    return(coefs_by_features)


# In[7]:


# symbol별 coefficient 저장용 데이터프레임 생성
features_list = parquets_by_symbol_id[0].columns[4: -9]
coefs_by_symbol = pl.DataFrame({"index": features_list})

#symbol별 coefficient 도출
for i in list(symbol_ids[0:3]) :
    print(f'symbol_ids: {i}')
    coefs_by_symbol = coefs_by_symbol.with_columns(pl.Series(name=f'coef_symbol{i}', values=coef(i)['coef']))

coefs_by_symbol


# In[8]:


fig = px.bar(x=coefs_by_symbol['index'], y=coefs_by_symbol['coef_symbol0'])
fig.update_layout(title_text="coef_symbol_id_0", title_x=0.5)
fig.show()


# In[9]:


px.bar(x=coefs_by_symbol['index'], y=coefs_by_symbol['coef_symbol1'])
fig.update_layout(title_text="coef_symbol_id_1", title_x=0.5)
fig.show()


# In[10]:


px.bar(x=coefs_by_symbol['index'], y=coefs_by_symbol['coef_symbol2'])
fig.update_layout(title_text="coef_symbol_id_2", title_x=0.5)
fig.show()


# In[11]:


print(coefs_by_symbol.top_k(30, by='coef_symbol1')["index"].to_list())


# In[12]:


def top30fig(symbol) :
    #symbol = 0
    coefs_by_symbol_target = coefs_by_symbol['index', f'coef_symbol{symbol}']
    top30 = coefs_by_symbol_target.top_k(30, by=f'coef_symbol{symbol}')['index']

    coefs_by_symbol_target
    coefs_by_symbol_target = coefs_by_symbol_target.with_columns(
        pl.when(pl.col('index').is_in(top30)).then(pl.lit('#1f77b4'))
        .otherwise(pl.lit('#7f7f7f'))#1f77b4
        .alias('color'))

    fig = go.Figure()
    colors = coefs_by_symbol_target['color'].to_list()

    fig.add_trace(go.Bar(x=coefs_by_symbol_target['index'],y=coefs_by_symbol_target[f'coef_symbol{symbol}'], marker_color=colors))
    fig.update_layout(title_text=f'coef_symbol_id_{symbol}, top30', title_x=0.5)
    fig.show()

    print(top30.to_list())


# In[13]:


top30fig(0)


# In[14]:


top30fig(1)


# In[15]:


top30fig(2)

