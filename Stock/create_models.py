# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:57:38 2022

@author: mvssa
"""
import glob
import pandas as pd
from preprocess import stock_preprocess as stck_proc
stocks_existing=[]
import random
stock_index=random.sample(range(len(glob.glob('FullDataCsv/*.csv'))),k=15)
all_stocks=glob.glob('FullDataCsv/*.csv')
stock_for_test=[all_stocks[i] for i in stock_index]
for models in glob.glob('Scalers and models/**/*.h5'):
    stocks_existing.append(models.split('\\')[1])
for file_name in stock_for_test:
    stock_name=file_name.split('\\')[1].split('.')[0].split('__')[0]
    print(f"Stock: {stock_name}")
    if stock_name in stocks_existing:
        print(f'{stock_name} Model already Exists, So not Training!!')
    else:
        df=pd.read_csv(file_name)
        df=df.set_index('timestamp')
        df.dropna(inplace=True)
        df.index=pd.to_datetime(df.index,dayfirst=True,errors='ignore').astype(int)/10**10
        
        
        df=df.dropna()
        time_period=60
        shift = -(2*time_period)+1
        df['close_avg'] = df['close'].rolling(window=time_period).mean().shift(shift)
        
        input_vars=['open','close']
        output_vars=['close_avg']
        model_types=['CNN','LSTM']
        
        stock_processor=stck_proc(stock_name, df, input_vars, output_vars,model_type='CNN')
        X,y=stock_processor.create_X_y(input_vars,output_vars)
        x_scaled,y_scaled,scaler_x,scaler_y=stock_processor.scale_data(X,y)
        import pickle
        try:
            pickle.dump(scaler_x,open(f'{stock_name}/{stock_name}_scaler_x.pkl','wb'))
            pickle.dump(scaler_y,open(f'TEST/{stock_name}_scaler_y.pkl','wb'))
        except FileNotFoundError:
            try:
                import os
                os.mkdir(f'Scalers and models/{stock_name}')
                pickle.dump(scaler_x,open(f'Scalers and models/{stock_name}/{stock_name}_scaler_x.pkl','wb'))
                pickle.dump(scaler_y,open(f'Scalers and models/{stock_name}/{stock_name}_scaler_y.pkl','wb'))
            except FileExistsError:
                print('Already Scalers Exist !!')
                pass
        
        percent_data=100
        
        features,targets=stock_processor.cnn_data_preprocess(x_scaled, y_scaled, percent_data, time_period)
        x_train, x_test, y_train, y_test=stock_processor.create_x_y_and_final_split(features, targets)
        from train import model_train
        trainer=model_train(x_train, x_test, y_train,y_test,stock_name,model_type=model_types[0],optimizer='SGD')
        model=trainer.train()
        trainer.evaluate(model,scaler_y)
        #trainer.log_in_csv()