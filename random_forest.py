# -*- coding: utf-8 -*-
"""
Created on Wed May  4 01:47:01 2016

@author: cweinshenker
"""

import numpy as np
import pandas as pd
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

#Import data
train = pd.read_csv("train_preprocessed.csv", header = 0, sep = ",")
test = pd.read_csv("test_preprocessed.csv", header = 0, sep = ",")
x_train = train.iloc[:,19:]
y_train = train.iloc[:, 17]
x_test = test.iloc[:,10:]

#Scale
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)


#Generate a random forest
#Cross validate the number of decision trees
n_estimators = np.arange(20, 200, 20)
parameters = {'n_estimators': n_estimators}
rf = RandomForestClassifier(n_jobs=2)
clf = sklearn.grid_search.GridSearchCV(rf, parameters, cv = 5)
clf.fit(x_train_std, y_train)
y_pred = clf.predict_proba(x_test_std)

#Label data appropriately and export to csv
data = {'Return_to_owner': y_pred[:,0],\
        'Euthanasia': y_pred[:,1],\
        'Died': y_pred[:,2], \
        'Adoption': y_pred[:,3],\
        'Transfer': y_pred[:, 4]}
results = pd.DataFrame(data = data)
results.to_csv("random_forest_results.csv")

