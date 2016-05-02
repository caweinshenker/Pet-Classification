# -*- coding: utf-8 -*-
"""
Created on Mon May  2 00:58:17 2016

@author: cweinshenker
"""


import pandas as pd

#More preprocessing
train = pd.read_csv("train_cleaned.csv", header = 0)
test = pd.read_csv("test_cleaned.csv", header = 0)

#Zero null values for the ages
train.AgeuponOutcome_Numeric[train.AgeuponOutcome_Numeric.isnull()] = 0
test.AgeuponOutcome_Numeric[test.AgeuponOutcome_Numeric.isnull()] = 0

#Get relevant feature matrices
x_train = train.iloc[:,13:]
y_train = train.iloc[:,11]
x_test = test.iloc[:,9:]


#Get breeds missed in the data cleaning file and add them to training
new_keys = []
for i in x_test.keys():
    if i not in x_train.keys():
        new_keys.append(i)
        train[i] = [0 for i in range(train.shape[0])]
      
#Assign the missing breeds      
for i in range(train.shape[0]):
    if train.Breed[i] in new_keys:
        new_breed = train.Breed[i]
        train.new_breed = 1
    
#Create dummy columns for outcome types        
for column in ['OutcomeType']: 
        dummies = pd.get_dummies(train[column])
        train[dummies.columns] = dummies

#Assign dummies for outcomes
for i in range(train.shape[0]):
    outcome = train.OutcomeType[i]
    train[outcome][i] = 1

        
        
train.to_csv("train_preprocessed.csv")
test.to_csv("test_preprocessed.csv")
    