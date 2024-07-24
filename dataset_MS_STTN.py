import numpy as np
import os
import pandas as pd
from pandas import DataFrame as df
import csv

import torch
import torch.utils.data as data

class CustomSensorDataset(data.Dataset):
    def __init__(self, user_names, data_path):

        self.x_data, self.y_data = self.load_sensor_data(user_names, data_path)
        
    def __len__(self):

        return len(self.x_data)
    
    def __getitem__(self, index):
        
        return self.x_data[index], self.y_data[index]
    
    def load_sensor_data(self, user_names, data_path):
        
        x, y = list(), list()
        
        for user_name in user_names:

            sensor_data_path = data_path + user_name + '/sensor/'
            label_path = data_path + user_name + '/label.csv'
            label_list = pd.read_csv(label_path,header=None)

            count = 0

            sensor_data_list = os.listdir(sensor_data_path)
            sensor_data = list()
            for file in sensor_data_list:
                f = os.path.join(sensor_data_path, file)
                sensor_data.append(f)

            for file in sensor_data:

                label = label_list.iloc[count].values.tolist()

                data = pd.read_csv(file)

                select_list = []
                for i in range(0, len(data), 3):
                    select_list.append(i)

                sensor = data.loc[select_list]
                sensor = sensor.drop('Second',axis=1)

                #25Hz
                segment1 = sensor.loc[:1124].to_numpy()
                x.append(segment1)
                y.append(int(label[0]))
                segment2 = sensor.loc[1125:2249].to_numpy()
                x.append(segment2)
                y.append(int(label[1]))
                segment3 = sensor.loc[2250:3374].to_numpy()
                x.append(segment3)
                y.append(int(label[2]))

                count += 1
        
        x = torch.FloatTensor(x)   
        y = torch.FloatTensor(y)
        
        return x, y