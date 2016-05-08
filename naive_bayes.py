# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:12:02 2016

@author: cweinshenker
This file implements a Naive Bayesian classifier approach to animal outcome 
classification.
Best Kaggle result using this method: 6.6924
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


#Run a Bernoulli Naive Bayes classifier on the data
#While we use the predict_proba method here, Naive Bayes is notorious
#as a poor predictor, so it's likely that these probabilities will be 
#way off
#In addition, Bernoulli Naive Bayes doesn't jive well with the fact that
#age is not a binary variable. It will be  binarized, though this is less
#harmful than it may seem, as the cutoff of a single year of age seems
#to be an important factor in determining outcome
bnb = BernoulliNB()
bnb.fit(x_train_pca, y_train)
y_pred = bnb.predict_proba(x_test_pca)

#Label data appropriately and export to csv
data = {'Return_to_owner': y_pred[:,0],\
        'Euthanasia': y_pred[:,1],\
        'Died': y_pred[:,2], \
        'Adoption': y_pred[:,3],\
        'Transfer': y_pred[:, 4]}
results = pd.DataFrame(data = data)
results.to_csv("naive_bayes_results.csv")

