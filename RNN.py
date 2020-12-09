# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 09:31:53 2020

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
plt.plot(Dnord.Volume)
plt.plot(Dsud.Volume)

"""
On va maintenant s'appuyer sur le RNN vu en cours
"""

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hiddensize =20
        self.n_layers = 1
        self.rnn = nn.RNN(1, self.hiddensize, self.n_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(self.hiddensize, 2)

        # special init: predict inputs
        # self.rnn.load_state_dict({'weight_ih_l0':torch.Tensor([[1.]]),'weight_hh_l0':torch.Tensor([[0.]]),'bias_ih_l0':torch.Tensor([0.]),'bias_hh_l0':torch.Tensor([0.])},strict=False)
        print(self.rnn.state_dict())
        # self.fc.load_state_dict({'weight':torch.Tensor([[1.]]),'bias':torch.Tensor([0.])},strict=False)
        print(self.fc.state_dict())

    def forward(self, x):

        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        _, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(hidden.view(-1,self.hiddensize))

        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hiddensize)
        return hidden
    
class ModelAtt(Model):
    def __init__(self):
        super(ModelAtt, self).__init__()
        qnp = 0.1*np.random.rand(self.hiddensize)
        self.q = nn.Parameter(torch.Tensor(qnp))

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        steps, last = self.rnn(x, hidden)
        alpha = torch.matmul(steps,self.q)
        alpha = nn.functional.softmax(alpha,dim=1)
        alpha2 = alpha.unsqueeze(-1).expand_as(steps)
        weighted = torch.mul(steps, alpha2)
        rep = weighted.sum(dim=1)
        out = self.fc(rep)
        return out, alpha


class ClassDataset(Dataset):
    def __init__(self, feature, target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, idx):
        item = self.feature[idx]
        label = self.target[idx]        
        return item,label

"""
On va chercher (pour exemple) à prédire le volume de traffic en direction du nord
"""

"""
la première chose est de créer une séquence de fenêtre (on va prendre 30 jours) sur laquelle 
se basera l'apprentissage
Notre série temporelle devient ainsi une liste de liste :
    [ [volumes mesurés du premier au 30ème jour]
      [volumes mesurés du deuxième au 31ème jour]
      [volumes mesurés du troisième au 32ème jour]
      ...
    ]
A laquelle on adjoindra les valeurs cibles :
    [ volume mesuré au 31ème jour
      volume mesuré au 32ème jour
      volume mesuré au 33ème jour
      ...
    ]
"""



def sequences(data, wind_size):
    x = []
    y = []
    for i in range(len(data)-wind_size):
        x.append(data[i:i+wind_size])
        y.append(data[i + wind_size])
    return np.array(x), np.array(y)
        
X, Y = sequences(Dnord.Volume, 30)

"""
On va maintenant utiliser les 2 premiers tiers de notre série temporelle pour servir
de base d'apprentissage et le dernier tiers de base de validation
"""

train_size = int(len(Y)*0.67)
test_size = len(Y) - train_size

X_train = X[train_size]
Y_train = Y[train_size]
X_val = X[test_size]
Y_val = Y[test_size]

train = ClassDataset(X_train.reshape(X_train.shape[0],1), Y_train)
valid = ClassDataset(X_val.reshape(X_val.shape[0],1), Y_val)
train_loader = DataLoader(train)
valid_loader = DataLoader(valid)





