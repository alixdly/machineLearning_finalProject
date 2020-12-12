#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:11:17 2020

@author: alixdelannoy,arthurmouchot
"""

#Librairies utiles
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Chargement des données
data=pd.read_csv('Radar_Traffic_Counts.csv')

#Informations sur les données
data.info()
data.describe().T #4 603 861 lignes dans nos données
data.isnull().sum()#pas de trous dans les données

def summary(df):
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nas = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing = (df.isnull().sum() / df.shape[0]) * 100
    print('Data shape:', df.shape)
    cols = ['Type', 'Total count', 'Null Values', 'Distinct Values', 'Missing Ratio', 'Unique Values']
    dtls = pd.concat([types, counts, nas, distincts, missing, uniques], axis=1, sort=False)
    dtls.columns = cols
    return dtls

infos=summary(data)

#Normaliser les données
traffic_volumes=np.array(data["Volume"])
scaler = MinMaxScaler(feature_range=(0, 1))
traffic_volumes_normalized = scaler.fit_transform(traffic_volumes.reshape(-1, 1))
data["Volume"] = traffic_volumes_normalized


#Fonction pour ouvrir les données en séléctionnant ce qu'on veut
#On somme les volumes de traffic par jour
def DataReader(csv,year=None,month=None, location=None, direction=None):
    #Ouvrir le csv et construire une colonne avec la date complète; Année+Mois+jour+TimeBin
    data = pd.read_csv(csv, parse_dates={"date": [3, 4, 5, 9]}, keep_date_col=True)
    #Select wanted year
    if year is not None:
        data = data.loc[data["Year"] == str(year)]
    #Select wanted month
    if month is not None:
        data = data.loc[data["Month"] == str(month)]
    #Select wanted location and traffic direction
    if location is not None:
        data = data.loc[data["location_name"] == location]
        data = data.loc[data["Direction"] == direction]
    data = data[['Year', 'Month', 'Day', 'Volume']]
    data=data.groupby(['Year', 'Month', 'Day'], as_index = False).sum()
    #Nomaliser
    traffic_volumes=np.array(data["Volume"])
    scaler = MinMaxScaler(feature_range=(0, 1))
    traffic_volumes_normalized = scaler.fit_transform(traffic_volumes.reshape(-1, 1))
    data["Volume"] = traffic_volumes_normalized
    return data

test=DataReader('Radar_Traffic_Counts.csv',location=' CAPITAL OF TEXAS HWY / LAKEWOOD DR',direction='NB')

#Fonction qui permet de séparer les données en 
def DataSplit(data, time_window):
    inout_seq = []
    L = len(data)
    for i in range(L-time_window):
        train_seq = data[i:i+time_window]
        train_label = data[i+time_window:i+time_window+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

test2=DataSplit(test.Volume,30)


