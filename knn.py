# -*- coding: utf-8 -*-
"""
Created on Mon May  2 00:01:20 2016

@author: cweinshenker
"""

#Clearly KNN will be an absolute flop
#It only outputs a nonzero probability for one class,
#while we should have a probability prediction for each class

#Results 16.09659

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


train = pd.read_csv("train_preprocessed.csv", header = 0)
test = pd.read_csv("test_preprocessed.csv", header = 0)
#Zero null values for the ages
train.AgeuponOutcome_Numeric[train.AgeuponOutcome_Numeric.isnull()] = 0
test.AgeuponOutcome_Numeric[test.AgeuponOutcome_Numeric.isnull()] = 0

x_train = train.iloc[:,14:]
y_train = train.iloc[:,12]
x_test = test.iloc[:,10:]
print(x_train.shape)
print(x_test.shape)

        

#Implement a KNN model using Euclidean distance metric
knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = "minkowski")

x_train.AgeuponOutcome_Numeric.isnull()
knn.fit(x_train, y_train)

#Predict
y_pred = knn.predict(x_test) #Predict the class labels for data
#Assign dummy labels based on following dictionary
#{'Return_to_owner': 0, 'Euthanasia': 1, 'Died': 4, 'Adoption': 2, 'Transfer': 3}
results_dict = {0: 'Return_to_owner', 1 : 'Euthanasia', 2: 'Adoption', 3: 'Transfer', 4 : 'Died'}
rows = [0 for i in range(x_test.shape[0])]
#'ID': [i for i in range(len(rows))], 
data = {'Return_to_owner': rows, 'Euthanasia': rows, 'Died': rows, 'Adoption': rows, 'Transfer': rows}
results = pd.DataFrame(data = data)
for i in range(len(y_pred)):
    result = results_dict[y_pred[i]]
    print(result)
    results[result][i] = 1
results.to_csv("knn_results.csv")
    

    
