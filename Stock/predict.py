# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:44:25 2022

@author: mvssa
"""
import pickle
import numpy as np
import glob
time_period=60
shift = -(2*time_period)+1
import pandas as pd
from tensorflow import keras
investments=dict()
returns=dict()
stock_map=dict()
for stocks in glob.glob('FullDataCsv/*.csv'):
    stock_name_=stocks.split('\\')[-1].split('__')[0]
    stock_map[f'{stock_name_}']=stocks
for model_path in glob.glob('Scalers and models/**/*.h5'):
    stock_name=model_path.split('\\')[1]
    print(f'Analysing STOCK: {stock_name}')
    df=pd.read_csv(stock_map[stock_name])
    df['close_avg'] = df['close'].rolling(window=time_period).mean().shift(shift)
    test_data=df[['open','close']].iloc[shift:shift+time_period].to_numpy()
    y_test=df[['close_avg']].iloc[shift].to_numpy()
    scaler_x=pickle.load(open(f'Scalers and models/{stock_name}/{stock_name}_scaler_x.pkl','rb'))
    scaler_y=pickle.load(open(f'Scalers and models/{stock_name}/{stock_name}_scaler_y.pkl','rb'))
    exp_data=np.reshape(scaler_x.transform(test_data),(1,time_period,2,1))
    model = keras.models.load_model(model_path)
    pred=scaler_y.inverse_transform(model.predict(exp_data))
    todays_val=df[['close']].iloc[shift+time_period-1].to_numpy()
    
    if pred-todays_val>0:
        investments[f'{stock_name}']=todays_val[0]
        returns[f'{stock_name}']=pred[0][0]
    else:
        pass

invest_in=list(investments.keys())
total_investments=sum(list(investments.values()))
total_returns=sum(list(returns.values()))
profit=total_returns-total_investments

print(f'Invest in {invest_in}')
print(f'Total Investment needed: {total_investments}')
print(f'Total Returns Predicted: {total_returns}')
print(f'Total PROFIT Expected: {profit}')