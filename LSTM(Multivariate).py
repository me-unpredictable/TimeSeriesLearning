# Author: me__unpredictable

# This is a basic example of LSTM network using multivariate data.

import pandas as pd
import numpy as np
import os
import wget

import torch
from torch.utils.data import Dataset, DataLoader

# Download and Extract Air quality dataset from UCI
# https://archive.ics.uci.edu/ml/datasets/Air+Quality
dataset_url = 'https://archive.ics.uci.edu/static/public/360/air+quality.zip'
if 'dataset' not in os.listdir('.'):
    os.mkdir('dataset')
    wget.download(dataset_url, out='dataset/data.zip')
    os.system('cd dataset && 7z x data.zip && rm data.zip')

# Now load the data using pandas
df = pd.read_csv('dataset/AirQualityUCI.csv', sep=';', decimal=',')
# drop na values
df.dropna(how='all', axis=1, inplace=True)
print(df.head())
print(df.columns.values)
# ------------------------------------------------------------------
# split data into training and testing sets
split_len = int(len(df) * 0.8)  # 80 percent data for the training and 20 percent for the testing
train_df = df[:split_len]
test_df = df[split_len:]

# -------------------------------------------------------------------
# normalize data
mean_data = train_df.mean()
std_data = train_df.std()
train_df = (train_df - mean_data) / std_data
test_df = (test_df - mean_data) / std_data

# ------------------------------------------------------------------

# Let's create a pytorch dataset from loaded data
class AirQuality(Dataset):
    def __init__(self,df):
        self.features=df[['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)',
                          'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
                          'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
                          'PT08.S5(O3)', 'T', 'RH']]
        self.target=df[['PT08.S5(O3)']]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()

        sample={'features': self.features.iloc[idx].values,
                'target':self.target.iloc[idx].values}

        return sample

# -----------------------------------------------------------------

# pytorch dataloader
train_data= AirQuality(train_df)
test_data= AirQuality(test_df)

batch_size=100

train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=batch_size,shuffle=True)


# -----------------------------------------------------------------
# train the LSTM model

import torch.optim as optim
import torch.nn as nn
import LSTM_multi_model as m

input_size=12
hidden_size=64
num_layers= 2
output_size=1
lr =0.001
epochs=100

model=m.MultiLSTM(input_size,hidden_size,num_layers,output_size)
model.to('cuda')
criterion=nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

for epoch in range(epochs):
    train_loss=0.0

    for i, data in enumerate(train_dataloader):
        features,targets = data['features'], data['target']
        features=features.float()
        features=features.to('cuda')
        targets=targets.float()
        targets=targets.to('cuda')

        optimizer.zero_grad(set_to_none=True)

        output=model(features)
        loss=criterion(output,targets)
        loss.backward()
        optimizer.step()

        train_loss+= loss.item() * features.size(0)
    train_loss /=len(train_dataloader.dataset)

    print('Epoch:',epoch+1,'Loss: ',train_loss)

# ---------------------------------------------------------------

# evaluating the model
test_loss=0.0

with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        features,targets=data['features'],data['target']
        features = features.float()
        features = features.to('cuda')
        targets = targets.float()
        targets = targets.to('cuda')

        output=model(features)
        print(output,targets)
        loss=criterion(output,targets)
        print(loss.item())
        test_loss +=loss.item() * features.size(0)

test_loss /= len(test_dataloader.dataset)

print('Test Loss:',test_loss)