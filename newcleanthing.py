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


Adopted = train["Adoption"]
Euthanasia = train['Euthanasia']
Age = train['AgeuponOutcome_Numeric']#inyears

ADOP_young = 0
EUTH_young = 0
ADOP_old = 0
EUTH_old = 0
old = 0
young = 0
YA = 0
ADOP_YA = 0
EUTH_YA = 0

for i in range(26729):
    if train.loc[i,'AgeuponOutcome_Numeric']>=3:
        old = old+1
    elif 3>train.loc[i,'AgeuponOutcome_Numeric']>=1:
        YA = YA + 1
    elif train.loc[i,'AgeuponOutcome_Numeric']<1:
        young = young + 1
        
print(old,YA,young)

for i in range(26729):
    if Adopted[i] == 1 and train.loc[i,'AgeuponOutcome_Numeric']>=3:
        ADOP_old = ADOP_old+1
    elif Adopted[i] == 1 and 3>train.loc[i,'AgeuponOutcome_Numeric']>=1:
        ADOP_YA = ADOP_YA + 1
    elif Adopted[i] == 1 and train.loc[i,'AgeuponOutcome_Numeric']<1:
        ADOP_young = ADOP_young + 1

print(ADOP_old/old, ADOP_YA/YA,ADOP_young/young)

for i in range(26729):
    if Euthanasia[i] == 1 and train.loc[i,'AgeuponOutcome_Numeric']>=3:
        EUTH_old = EUTH_old+1
    elif Euthanasia[i] == 1 and 3>train.loc[i,'AgeuponOutcome_Numeric']>=1:
        EUTH_YA = EUTH_YA + 1
    elif Euthanasia[i] == 1 and train.loc[i,'AgeuponOutcome_Numeric']<1:
        EUTH_young = EUTH_young + 1

print(EUTH_old/old, EUTH_YA/YA,EUTH_young/young)



            


    


