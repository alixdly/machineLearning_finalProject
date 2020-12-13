#Implémentation de la correction du projet "Predict Future Sales"

#Librairies utiles
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from finalProject import DataReader

data=DataReader('Radar_Traffic_Counts.csv',year=2018, month=2,location=' CAPITAL OF TEXAS HWY / LAKEWOOD DR',direction='NB')
train_set = data[:'2018-02-07 00:00:00']
valid_set = data['2018-02-07 00:00:00':'2018-02-08 00:00:00']

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

n_steps = 3
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
        print(label.shape)
        
        return item,label

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1d = nn.Conv1d(3,64,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64*2,50)
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
model=CNN()
lr=0.01
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
criterion=nn.MSELoss()

train = RadarDataset(train_x.reshape(train_x.shape[0],train_x.shape[1],1),train_y)
valid = RadarDataset(valid_x.reshape(valid_x.shape[0],valid_x.shape[1],1),valid_y)
train_loader = DataLoader(train,batch_size=2,shuffle=False)
valid_loader = DataLoader(train,batch_size=2,shuffle=False)

train_losses = []
valid_losses = []

def Train():
    
    running_loss = .0
    model.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = criterion(preds,labels)
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
            loss = criterion(preds,labels)
            running_loss += loss
            
        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.detach().numpy())
        print(f'valid_loss {valid_loss}')
        
epochs = 200
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


target_x , target_y = split_sequence(train_set.Volume.values,n_steps)
inputs = target_x.reshape(target_x.shape[0],target_x.shape[1],1)

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