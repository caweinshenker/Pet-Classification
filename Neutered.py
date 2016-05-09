# -*- coding: utf-8 -*-
"""
Created on Sun May  8 17:29:52 2016

@author: jonathanrigby
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:13:39 2016

@author: jonathanrigby
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:12:33 2016

@author: jonathanrigby
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import date
import calendar
from time import *
from datetime import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#pd.dummies

train_filename = 'train_preprocessed.csv'
train = pd.read_csv(train_filename)

Adoption = train["Adoption"]
Euthanasia = train['Euthanasia']

Neutered = train['Neutered Male']
Spayed = train['Spayed Female']
Not_Neutered = train['Intact Male']
Not_Spayed = train['Intact Female']
unknown = train['Unknown']

NNEUT = 0
NSPAY = 0
SPAY = 0
NEUT = 0
UNK = 0

ADOP_N = 0
ADOP_S = 0
ADOP_NN = 0
ADOP_NS = 0
ADOP_UNK = 0

EUTH_N = 0
EUTH_S = 0
EUTH_NN = 0
EUTH_NS = 0
EUTH_UNK = 0


for i in range(26729):
    if train.loc[i,'Spayed Female']==1:
        SPAY = SPAY+1
    elif train.loc[i,'Neutered Male']==1:
        NEUT = NEUT + 1
    elif train.loc[i,'Intact Male']==1:
        NNEUT = NNEUT + 1
    elif train.loc[i,'Intact Female']==1:
        NSPAY = NSPAY + 1

        
        
print(SPAY,NSPAY,NEUT,NNEUT)

Female = SPAY+NSPAY
Male = NEUT+NNEUT

for i in range(26729):
    if Adoption[i]==1 and train.loc[i,'Spayed Female']==1:
        ADOP_S = ADOP_S+1
    elif Adoption[i]==1 and train.loc[i,'Neutered Male']==1:
        ADOP_N = ADOP_N + 1
    elif Adoption[i]==1 and train.loc[i,'Intact Male']==1:
        ADOP_NN = ADOP_NN + 1
    elif Adoption[i]==1 and train.loc[i,'Intact Female']==1:
        ADOP_NS = ADOP_NS + 1


print(ADOP_S/SPAY,ADOP_NS/NSPAY, ADOP_N/NEUT,ADOP_NN/NNEUT)

for i in range(26729):
    if Euthanasia[i]==1 and train.loc[i,'Spayed Female']==1:
        EUTH_S = EUTH_S+1
    elif Euthanasia[i]==1 and train.loc[i,'Neutered Male']==1:
        EUTH_N = EUTH_N + 1
    elif Euthanasia[i]==1 and train.loc[i,'Intact Male']==1:
        EUTH_NN = EUTH_NN + 1
    elif Euthanasia[i]==1 and train.loc[i,'Intact Female']==1:
        EUTH_NS = EUTH_NS + 1


print(EUTH_S/SPAY,EUTH_NS/NSPAY,EUTH_N/NEUT,EUTH_NN/NNEUT)
    


