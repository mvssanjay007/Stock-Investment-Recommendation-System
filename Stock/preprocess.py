# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:48:28 2022

@author: mvssa
"""
import numpy as np
from sklearn.model_selection import train_test_split
class stock_preprocess:
    UP = 0
    DOWN = 1
    STATIONARY = 2
    def __init__(self,stock_name,df,input_vars,output_vars,model_type='CNN'):
        
        self.stock_name=stock_name
        self.input_vars=input_vars
        self.output_vars=output_vars
        self.df=df
        self.model_type=model_type
    
    def create_X_y(self,input_vars,output_vars):
        self.df=self.df.dropna()
        data1=self.df[input_vars+output_vars]
        X=data1[input_vars].to_numpy()
        y=data1[output_vars].to_numpy()
        return X,y
    
    def scale_data(self,X,y):
            
        
            
            x_train_, x_test_, y_train_, y_test_ = train_test_split(X, y,shuffle=False,test_size=0.2)
            from sklearn.preprocessing import StandardScaler#,MinMaxScaler
            scaler_x=StandardScaler()
            scaler_y=StandardScaler()
            scaler_x.fit(x_train_)
            scaler_y.fit(y_train_)
            x_train_scaled=scaler_x.transform(x_train_)
            x_test_scaled=scaler_x.transform(x_test_)
            
            y_train_scaled=scaler_y.transform(y_train_)
            y_test_scaled=scaler_y.transform(y_test_)
            
            x_scaled=np.append(x_train_scaled,x_test_scaled,axis=0)
            y_scaled=np.append(y_train_scaled,y_test_scaled,axis=0)
            
            return x_scaled,y_scaled,scaler_x,scaler_y
        
    def cnn_data_preprocess(self,x_scaled,y_scaled,percent_data,time_period):
        features=[]
        targets=[]
        for i in range(int(x_scaled.shape[0]*(percent_data/100))-time_period):
            features.append(x_scaled[i:i+time_period])
            targets.append(y_scaled[i])
        return np.asarray(features),np.asarray(targets)
    
    def create_x_y_and_final_split(self,features,targets):
            X=features
            y=targets
            print(f'Data Processed for {self.stock_name}')
            if self.model_type=='CNN':        
                X=np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))
            else:
                X=np.reshape(X,(X.shape[0],X.shape[1],1))
            x_train, x_test, y_train, y_test = train_test_split(X, y,shuffle=False,test_size=0.2)
            print(f'Data Splitted for {self.stock_name} Model training')
            return x_train, x_test, y_train, y_test
    def label_data(self,row):
        stationary_threshold = .0001
        if row['close_avg_change_pct'] > stationary_threshold:
            return self.UP
        elif row['close_avg_change_pct'] < -stationary_threshold:
            return self.DOWN
        else:
            return self.STATIONARY
