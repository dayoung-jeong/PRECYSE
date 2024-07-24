import numpy as np
import os
import pandas as pd
from pandas import DataFrame as df
import csv
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
from timm.data import Mixup
from timm.models import create_model
from timm.utils import NativeScaler, get_state_dict, ModelEma#, accuracy
from functools import partial
from einops import rearrange

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler

import utils as utils
from dataset_MS_STTS import CustomSensorDataset
from model_MS_STTS import SensorTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = './Data/'
user_number = os.listdir(data_path)

dataset = CustomSensorDataset(user_number, data_path)
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
test_size = dataset_size - train_size
train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])
test_dataloader = data.DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)

print(f"Training Data Size : {len(train_dataset)}")
print(f"Testing Data Size : {len(test_dataset)}")

k = 5
kfold = KFold(n_splits=k, shuffle=True)

model = SensorTransformer()
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 500

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
    
    print(f'Fold: {fold}')
    
    train_subsampler = data.SubsetRandomSampler(train_idx)
    val_subsampler = data.SubsetRandomSampler(val_idx)
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=16, sampler=train_subsampler)
    val_dataloader = data.DataLoader(train_dataset, batch_size=4, sampler=val_subsampler)
    
    for epoch in range(1,epochs+1):

        model.train()
        
        train_mae = 0
        train_rmse = 0
        
        for inputs, targets in train_dataloader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            
            ground_truth = torch.argmax(outputs, 1)

            batch_loss = criterion(outputs, targets.long())
            
            batch_loss.backward()
            optimizer.step()
            
            y_true = targets.cpu().detach().numpy()
            y_pred = ground_truth.cpu().detach().numpy()
            
            train_mae = mean_absolute_error(y_true,y_pred)
            train_rmse = np.sqrt(mean_squared_error(y_true,y_pred))
            
            train_mae += train_mae
            train_rmse += train_rmse

        train_mae = train_mae / 16
        train_rmse = train_rmse / 16
        
        model.eval()
        
        test_mae = 0
        test_rmse = 0

        with torch.no_grad():
            
            for inputs, targets in val_dataloader:

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                ground_truth = torch.argmax(outputs, 1) 

                batch_loss = criterion(outputs, targets.long())
                
                y_true = targets.cpu().detach().numpy()
                y_pred = ground_truth.cpu().detach().numpy()

                test_mae = mean_absolute_error(y_true,y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_true,y_pred))

                test_mae += test_mae
                test_rmse += test_rmse
        
        test_mae = test_mae / 4
        test_rmse = test_rmse / 4
                
        epoch_log = "Epoch:{}/{}".format(epoch,epochs)
        
        train_log = "AVG Training MAE:{:.5f} AVG Training RMSE:{:.5f}".format(train_mae,train_rmse)
        test_log = "AVG Test MAE:{:.5f} AVG Test RMSE:{:.5f}".format(test_mae,test_rmse)
        
        log = epoch_log + ' | ' + train_log + ' | ' + test_log
        
        print(log)

avg_train_mae = np.mean(history_train['MAE'])
avg_train_rmse = np.mean(history_train['RMSE'])

avg_test_mae = np.mean(history_test['MAE'])
avg_test_rmse = np.mean(history_test['RMSE'])

print('Performance of {} fold cross validation'.format(k))

print("AVG Training MAE:{:.5f} AVG Training RMSE:{:.5f}"
      .format(avg_train_mae,avg_train_rmse))
print("AVG Test MAE:{:.5f} AVG Test RMSE:{:.5f} "
      .format(avg_test_mae,avg_test_rmse))