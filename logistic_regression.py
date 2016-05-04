# -*- coding: utf-8 -*-
"""
Created on Tue May  3 22:01:48 2016

@author: cweinshenker
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.grid_search
from sklearn.linear_model import LogisticRegression

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


#Run logistic regression--get probabilities as opposed to classifications
#Cross validate C values
C = [1, 10, 100, 1000]
parameters = {'C': C}
lr = LogisticRegression(random_state = 0)
clf = sklearn.grid_search.GridSearchCV(lr, parameters, cv = 3)
clf.fit(x_train_std, y_train)
#lr.fit(x_train_std, y_train)
y_pred = clf.predict_proba(x_test_std)

#Label data appropriately and export to csv
data = {'Return_to_owner': y_pred[:,0],\
        'Euthanasia': y_pred[:,1],\
        'Died': y_pred[:,2], \
        'Adoption': y_pred[:,3],\
        'Transfer': y_pred[:, 4]}
results = pd.DataFrame(data = data)
results.to_csv("Logistic_regression_results.csv")
