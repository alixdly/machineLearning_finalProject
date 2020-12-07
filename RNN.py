# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 09:31:53 2020

@author: arthu
"""

import numpy as np
import torch
import torch.nn as nn
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









