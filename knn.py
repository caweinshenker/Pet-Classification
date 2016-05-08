# -*- coding: utf-8 -*-
"""
Created on Mon May  2 00:01:20 2016

@author: cweinshenker
"""

#Clearly KNN will be an absolute flop
#It only outputs a nonzero probability for one class,
#while we should have a probability prediction for each class

#Results 15.174

import numpy as np
import pandas as pd
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.grid_search
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train = pd.read_csv("train_preprocessed.csv", header = 0)
test = pd.read_csv("test_preprocessed.csv", header = 0)

#Exclude date
x_train = train.iloc[:,20:]
y_train = train.iloc[:,17]
x_test = test.iloc[:,11:]


#Scale 
sc = StandardScaler()
sc.fit(x_train) #Calculates s and x bar
x_train_std = sc.transform(x_train) #Performs normalization
x_test_std = sc.transform(x_test) #Performs normalization
 
# Now perform PCA
#Get the covariance matrix 
cov_mat = np.cov(x_train_std.T)

#Eigenvalues, eigenvectors
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

#Compute the variance explained ratio
total=sum(eigen_vals)
var_exp = eigen_vals/total
cum_var_exp= np.cumsum(var_exp) #calculate cum sum of explained variance

#Sort the eigenpairs by decreating order of eigenvalues
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True, key = (lambda x: x[0]))

#collect eigenvectors that correspond to 95% variance
ninetyfiveIndex =  np.nonzero(cum_var_exp > 0.95)[0][0] 
pca = PCA(n_components = ninetyfiveIndex)
pca.fit(x_train_std)
x_train_pca = pca.transform(x_train_std)
x_test_pca = pca.transform(x_test_std)

#Implement a KNN model 
#Use grid search to cross validate to find best k
#3 folds
#NOTE: This can take a LONG time
k = np.arange(4) + 1
parameters = {'n_neighbors': k}
knn = KNeighborsClassifier(p=2, metric = 'minkowski')
clf = sklearn.grid_search.GridSearchCV(knn, parameters, cv = 3)
clf.fit(x_train_pca, y_train)



#Predict
y_pred = clf.predict(x_test_pca)


#Assign dummy labels based on following dictionary
#{'Return_to_owner': 0, 'Euthanasia': 1, 'Died': 4, \
# 'Adoption': 2, 'Transfer': 3}
results_dict = {0: 'Return_to_owner', \
                1 : 'Euthanasia', \
                2: 'Adoption', \
                3: 'Transfer', 4 : 'Died'}
rows = [0 for i in range(x_test.shape[0])]
#'ID': [i for i in range(len(rows))], 
data = {'Return_to_owner': rows, 'Euthanasia': rows, \
        'Died': rows, 'Adoption': rows, 'Transfer': rows}
results = pd.DataFrame(data = data)
for i in range(len(y_pred)):
    result = results_dict[y_pred[i]]
    results[result][i] = 1
results.to_csv("knn_results.csv")
    

    
