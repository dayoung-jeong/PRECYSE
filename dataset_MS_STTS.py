import numpy as np
import os
import pandas as pd
from pandas import DataFrame as df
import csv

import torch
import torch.utils.data as data

class CustomSensorDataset(data.Dataset):
    def __init__(self, user_names, data_path):

        self.x_data, self.y_data = self.load_sensor_data(user_names,data_path)
        
    def __len__(self):

        return len(self.x_data)
    
    def __getitem__(self, index):
        
        return self.x_data[index], self.y_data[index]
    
    def load_sensor_data(self, user_names,data_path):
        
        x, y = list(), list()
        
        for user_name in user_names:
            
            spec_data_path = data_path + user_name + '/spectrogram/'
            label_path = data_path + user_name + '/label.csv'
            label_list = pd.read_csv(label_path,header=None)

            feature_count = 0
            index = 0
            label_count = 0
            segment_count = 0

            img_list= os.listdir(spec_data_path)

            img_data = [0 for i in range(32)]

            for img_name in img_list:

                if len(img_name) == 12:
                    index = int(img_name.split('.')[0][-1])
                elif len(img_name) == 13:
                    index = int(img_name.split('.')[0][-2:])

                img = Image.open(spec_data_path + img_name)
                crop_img=img.crop([55,36,389,252])
                resize_img = crop_img.resize((16,16))

                img_data[index] = np.array(resize_img.convert('RGB'))

                feature_count += 1

                if feature_count == 32:

                    label = label_list.iloc[label_count].values.tolist()

                    segment_count += 1

                    if segment_count == 1:
                        x.append(img_data)
                        y.append(int(label[0]))
                    elif segment_count == 2:
                        x.append(img_data)
                        y.append(int(label[1]))
                    elif segment_count == 3:
                        x.append(img_data)
                        y.append(int(label[2]))

                        segment_count = 0
                        label_count += 1

                    img_data = [0 for i in range(32)]
                    feature_count = 0
        
        x = torch.FloatTensor(x)   
        y = torch.FloatTensor(y)
        
        return x, y