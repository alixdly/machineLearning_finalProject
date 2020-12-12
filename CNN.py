#Impl�mentation de la correction du projet "Predict Future Sales"

#Librairies utiles
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Chargement des donn�es
data=pd.read_csv('Radar_Traffic_Counts.csv')

#Normaliser les donn�es
traffic_volumes=np.array(data["Volume"])
scaler = MinMaxScaler(feature_range=(0, 1))
traffic_volumes_normalized = scaler.fit_transform(traffic_volumes.reshape(-1, 1))
data["Volume"] = traffic_volumes_normalized

#On se concentre sur une localisation : CAPITAL OF TEXAS HWY / LAKEWOOD DR
D2 = data.loc[data.location_name == ' CAPITAL OF TEXAS HWY / LAKEWOOD DR']
D2.to_csv('Traffic_CapitalOfTexasHWY.csv', index=False, encoding='utf-8')
#Pour cette localisation, on a deux directions : North Bound et South Bound

dsi2c = {} #dictionaire des ventes par (shop,item)
lasthour=-1
#on ouvre le fichier et on r�cup�re les infos qui nous int�resse
with open("Traffic_CapitalOfTexasHWY.csv","r") as f:
    for l in f: break
    for l in f:
        c=l.split(",")
        hour = int(c[1])
        if date>lastmonth: lastmonth=date
        shop = int(c[2])
        item = int(c[3])
        count = float(c[-1])
        key = (shop,item)
        seq = dsi2c[key] if key in dsi2c else []
        seq += [(date,count)]
        dsi2c[key] = seq
print("nsequences %d %d" % (len(dsi2c.keys()),lastmonth))


import torch
import torch.nn as nn
from random import shuffle

xmin, xmax = 100.0, -100.0
vnorm = 1000.0
minlen = 8
# if <8 then layer1 outputs L=7/2=3 which fails because layer2 needs L>=4
si2X, si2Y = {}, {}
dsi2X, dsi2Y = {}, {}
for s,i in dsi2c.keys():
    seq = dsi2c[(s,i)]
    if len(seq)<=minlen: continue #passer � l'it�ration suivante, ne pas prendre en compte ce couple 
    xlist, ylist = [], []
    for m in range(minlen, len(seq)-1):
        xx = [seq[z][1]/vnorm for z in range(m)]
        if max(xx)>xmax: xmax=max(xx)
        if min(xx)<xmin: xmin=min(xx)
        xlist.append(torch.tensor(xx,dtype=torch.float32))
        yy = [seq[m+1][1]/vnorm]
        ylist.append(torch.tensor(yy,dtype=torch.float32))
    si2X[(s,i)] = xlist
    si2Y[(s,i)] = ylist
    if True: # build evaluation dataset
        xx = [seq[z][1]/vnorm for z in range(len(seq)-1)]
        dsi2X[(s,i)] = [torch.tensor(xx,dtype=torch.float32)]
        yy = [seq[len(seq)-1][1]/vnorm]
        dsi2Y[(s,i)] = [torch.tensor(yy,dtype=torch.float32)]

print("ntrain %d %f %f" % (len(si2X),xmin,xmax))

class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8)
        )
        self.fc1 = nn.Linear(in_features=64*8, out_features=128)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
 
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


mod = TimeCNN()
loss = torch.nn.MSELoss()
opt = torch.optim.Adam(mod.parameters(),lr=0.01)
for s, i in si2X.keys():
    xlist = si2X[(s,i)]
    if len(xlist)<10: continue
    ylist = si2Y[(s,i)]
    idxtr = list(range(len(xlist)))
    for ep in range(5):
        shuffle(idxtr)
        lotot=0.
        mod.train()
        for j in idxtr:
            opt.zero_grad()
            haty = mod(xlist[j].view(1,1,-1))
            # print("pred %f" % (haty.item()*vnorm))
            lo = loss(haty,ylist[j].view(1,-1))
            lotot += lo.item()
            lo.backward()
            opt.step()

        # the MSE here is computed on a single sample: so it's highly variable !
        # to make sense of it, you should average it over at least 1000 (s,i) points
        mod.eval()
        haty = mod(dsi2X[(s,i)][0].view(1,1,-1))
        lo = loss(haty,dsi2Y[(s,i)][0].view(1,-1))
        print("epoch %d loss %1.9f testMSE %1.9f" % (ep, lotot, lo.item()))