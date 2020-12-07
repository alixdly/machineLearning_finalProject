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

D = data.loc[rd.choices(data.index, k = 100)]



