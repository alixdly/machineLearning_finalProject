#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:11:17 2020

@author: alixdelannoy,arthurmouchot
"""

#Librairies utiles
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


#Répertoire de travail
os.getcwd()
os.chdir('/Users/alixdelannoy/Desktop/3A/MachineLearning/finalProject')

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





