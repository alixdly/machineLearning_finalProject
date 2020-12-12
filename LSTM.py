# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:29:20 2020

@author: arthu
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('Radar_Traffic_Counts.csv')

data.info()
data.describe().T #4 603 861 lignes dans nos données
data.isnull().sum()#pas de trous dans les données

D = data.loc[rd.choices(data.index, k = 1000)]#jeu d'exemple

"""Une première version naive va être de prédire de manière indépendante
le volume de traffic à un endroit donné en considérant que le traffic à un endroit
est indépendant du traffic qui peut exister ailleurs (ce qui est bien sur faux).
Par exemple, on va essayer de prédire le traffic au point :
    1612 BLK S LAMAR BLVD (Collier)
et pour les quatres directions possibles.
"""

D2 = data.loc[data.location_name == '1612 BLK S LAMAR BLVD (Collier)']
Dvisual = D2.loc[rd.choices(D2.index, k = 100)]#on isole juste 100 lignes pour la visualisation des données

#on continue notre approche naive ici en considérant indépendament les deux directions
Dnord = D2.loc[D2.Direction == 'NB']
Dsud = D2.loc[D2.Direction == 'SB']

#toujours dans une une approche naive, on ne considère que les volumes journaliers :
#On va donc sommer les volumes mesurés par jours
Dnord = Dnord[['Year', 'Month', 'Day', 'Volume']]
Dsud =Dsud[['Year', 'Month', 'Day', 'Volume']]

Dnord = Dnord.groupby(['Year', 'Month', 'Day'], as_index = False).sum()
Dsud = Dsud.groupby(['Year', 'Month', 'Day'], as_index = False).sum()

#on peut maintenant visualiser la série temporelle sur laquelle on va réaliser une prédiction grâce à un RNN
#plt.plot(Dnord.Volume)
#plt.plot(Dsud.Volume)

all_data = Dnord.Volume.values

"""
On dipose maintenant des donénes au bon format
Nous allons à présent séparer notre jeu de donénes en un jeu de d'apprentissage
et un jeu de test.
"""



test_data_size = int(len(all_data) /3)

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data = scaler.fit_transform(train_data .reshape(-1, 1))
test_data = scaler.fit_transform(test_data.reshape(-1, 1))

train_data = torch.FloatTensor(train_data).view(-1)
test_data = torch.FloatTensor(test_data).view(-1)

print(len(all_data))
print(len(train_data))
print(len(test_data))


train_window = 30

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data, train_window)
test_inout_seq = create_inout_sequences(train_data, train_window)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=10, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def Train():
    running_loss = 0
    model.train()
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        preds = model(seq)
        
        loss = loss_function(preds, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
    
    train_loss = running_loss/len(train_inout_seq)
    train_losses.append(train_loss.detach().numpy())
    if i%25 == 0:
        print("epoch : ", i, "    loss : ", train_loss)
    
    #print(f'train_loss {train_loss}')
    
def Valid():
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

            preds = model(seq)
            
            loss = loss_function(preds,labels)
            running_loss += loss
            
        valid_loss = running_loss/len(test_inout_seq)
        valid_losses.append(valid_loss.detach().numpy())
        #print(f'valid_loss {valid_loss}')
    
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 250
train_losses = []
valid_losses = []

for i in range(epochs):
    Train()
    Valid()

fut_pred = 30

test_inputs = train_data[-train_window:].tolist()
print(test_inputs)

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())
        
print(test_inputs[fut_pred:])

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
print(actual_predictions)

x = np.arange(len(Dnord.Volume)-train_window, len(Dnord.Volume), 1)
plt.plot(all_data)
plt.plot(x, actual_predictions)

plt.figure()
plt.plot(train_losses)
 


