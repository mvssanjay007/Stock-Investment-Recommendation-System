# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 17:06:04 2022

@author: mvssa
"""

import numpy as np
import pandas as pd
from preprocess import stock_preprocess
import glob
import pandas as pd
import numpy as np
import glob
import time
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D, MaxPooling2D,Activation, Flatten,LSTM
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tensorflow.keras.optimizers import Adam,SGD
total_start_time=datetime.datetime.now()
from sklearn.model_selection import train_test_split

class model_train(stock_preprocess):
    
    def __init__(self,x_train, x_test, y_train, y_test,stock_name,model_type='CNN',optimizer='SGD',time_period=60,filename='TimeTrack.csv'):
        self.filename=filename
        self.time_period=time_period
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.model_type=model_type
        self.optimizer=optimizer
        self.stock_name=stock_name
        
    def train(self,loss='mae',metrics=['mae'],batch_size=100,validation_split=0.2):        
            
            
            if self.model_type=='CNN':
                model=self.create_cnn(self.x_train.shape,self.y_train.shape)
            elif self.model_type=='LSTM':
                model=self.create_lstm(self.x_train.shape,self.y_train.shape)
            if self.optimizer=='Adam':
                opt=Adam(learning_rate=0.01)
            elif self.optimizer=='SGD':
                opt = SGD(learning_rate=0.01, momentum=0.9)
            model.compile(loss=loss,
                          optimizer=opt,
                          metrics=metrics)
            log_dir = "Logs/" +self.stock_name+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            start_time=time.time()
            print(f'{self.stock_name}_{self.model_type} Model Training Started')
            model.fit(self.x_train, self.y_train, batch_size=batch_size,validation_split=validation_split,verbose=1, epochs=15,callbacks=[tensorboard_callback])
            #model.fit(x_train, y_train, batch_size=100,validation_split=0.2, epochs=15)#,callbacks=[tensorboard_callback])
            end_time=time.time()
            print(f'{self.stock_name}_{self.model_type}  Model Training Done')
            self.time_taken=end_time-start_time
            return model
        
    
        

    def create_cnn(self,x_shape,y_shape):
            
        model = Sequential()
            
        model.add(Conv2D(64, (2, 2), input_shape=x_shape[1:]))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(y_shape[-1]))
        model.add(Activation('linear'))
        return model

    def create_lstm(self,x_shape,y_shape):
        
        model = Sequential()
        model.add(LSTM(units=50,input_shape=x_shape[1:],return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=50))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(y_shape[-1]))
        model.add(Activation('linear'))
        #model.add(Activation('softmax'))
        return model

    def log_in_csv(self):
        field_names=['Stock Name','Time_trained','Model Type','Time Taken to Train','Number of Stocks']
        with open(self.filename,'w') as csvfile:
            csvwriter=csv.writer(csvfile)
            #csvwriter.writerow(field_names)
            rows=[[self.stock_name,self.time,self.model_type,self.time_taken,self.y_train.shape[0]+self.y_test.shape[0]]]
            csvwriter.writerows(rows)
            
        
    def evaluate(self,model,scaler_y):
        
        
        
        y_pred=model.predict(self.x_test)
        x_axis=np.arange(self.x_test.shape[0])
        y_axis=scaler_y.inverse_transform(y_pred)-scaler_y.inverse_transform(self.y_test)
        
        print(f'{self.stock_name}_{self.model_type} Model evaluated')
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(x_axis,y_axis)
        plt.xlim([0,self.x_test.shape[0]])
        plt.xlabel('Stocks')
        plt.ylabel('Diff in Pred and True')
        plt.title(self.stock_name+self.model_type)
        plt.savefig(f'Scalers and models/{self.stock_name}/{self.stock_name}_{self.model_type}.png')
        
        print(f'Figure_saved_{self.stock_name}_{self.model_type}')
        self.time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path=f'Scalers and models/{self.stock_name}/{self.stock_name}_{self.model_type}_'+self.time+'.h5'
        model.save(model_path)
        print(f'Model_Saved_{self.stock_name}_{self.model_type}')
