#Librairies utiles
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from finalProject import DataReader


#Importer les données
data=DataReader('Radar_Traffic_Counts.csv',year=2018,location=' CAPITAL OF TEXAS HWY / LAKEWOOD DR',direction='NB')
train_set = data[:'2018-03-01 00:00:00']
valid_set = data['2018-03-01 00:00:00':'2018-03-10 00:00:00']

def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

n_steps = 12
train_x,train_y = split_sequence(train_set.Volume.values,n_steps)
valid_x,valid_y = split_sequence(valid_set.Volume.values,n_steps)

class RadarDataset(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label

train = RadarDataset(train_x.reshape(train_x.shape[0],1,train_x.shape[1]),train_y)
valid = RadarDataset(valid_x.reshape(valid_x.shape[0],1,valid_x.shape[1]),valid_y)
train_loader = DataLoader(train,batch_size=3,shuffle=False)
valid_loader = DataLoader(train,batch_size=3,shuffle=False)

#CNN : Convolutional Neural Networl pour données à 1 dimension avec 2 couches de convolution 

#Mon input évolue de la façon suivante :
#Input - Size = 12, Channels = 1
#First Layer
#Conv1D - Size = 10, Channels = 6
#ReLU - Unchanged
#MaxPool - Size = 6, Channels = 6
#Second Layer
#Conv1D - Size = 4, Channels = 12
#ReLU - Unchanged
#MaxPool - Size = 2, Channels = 6
#Full Connection Layer - output= 1 node
    
class CNN(nn.Module):

    #Layers of the network
    def __init__(self):
        super(CNN, self).__init__()
        #First convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 6, kernel_size = 3),#Convolution
            nn.ReLU(),#Activation non linéaire
            nn.MaxPool1d(kernel_size=2, stride=2) #Max Pooling layer
        )
        #Second convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels = 6, out_channels = 12, kernel_size = 3), #Convolution 
            nn.ReLU(), #Activation non linéaire
            nn.MaxPool1d(kernel_size=2, stride=2) #Max Pooling
        )
        self.fc = nn.Linear(in_features=2*6, out_features=1) #Full connection layer
        
    #Forward pass method
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
model=CNN()
lr=0.001
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
criterion=nn.MSELoss()

train_losses = []
valid_losses = []

def Train():
    running_loss = .0
    model.train()
    for idx, (inputs,labels) in enumerate(train_loader):
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = criterion(preds,labels.float().reshape(labels.shape[0],1))
        loss.backward()
        optimizer.step()
        running_loss += loss   
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().numpy()) 
    print(f'train_loss {train_loss}')
    
def Valid():
    running_loss = .0
    model.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds,labels.float().reshape(labels.shape[0],1))
            running_loss += loss
        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.detach().numpy())
        print(f'valid_loss {valid_loss}')

        
epochs = 100
for epoch in range(epochs):
    print('epochs {}/{}'.format(epoch+1,epochs))
    Train()
    Valid()

import matplotlib.pyplot as plt
plt.plot(train_losses,label='train_loss')
plt.plot(valid_losses,label='valid_loss')
plt.title('MSE Loss')
plt.ylim(0, 100)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


#Tracer la prediction
target_x,target_y = split_sequence(train_set.Volume.values,n_steps)
inputs=target_x.reshape(target_x.shape[0],1,target_x.shape[1])

model.eval()
prediction = []
batch_size = 2
iterations =  int(inputs.shape[0]/2)
for i in range(iterations):
    preds = model(torch.tensor(inputs[batch_size*i:batch_size*(i+1)]).float())
    prediction.append(preds.detach().numpy())
fig, ax = plt.subplots(1, 2,figsize=(11,4))
ax[0].set_title('predicted one')
ax[0].plot(prediction)
ax[1].set_title('real one')
ax[1].plot(target_y)
plt.show()