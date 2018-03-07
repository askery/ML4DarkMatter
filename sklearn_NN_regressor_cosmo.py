#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:40:58 2018

@author: askery
"""

from datetime import datetime
start=datetime.now()

import numpy as np
# regressors
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
# preprocess
from sklearn.model_selection import train_test_split #,cross_val_score,StratifiedKFold,cross_val_predict
from sklearn.decomposition import PCA
# metrics
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error
#from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

file1 = '/home/data/research/IIP/ML/colab/luciano/pedro/rawdata/data_all_algorithm.csv'
data = np.genfromtxt(file1,  delimiter=',', skip_header=1) # load data
X = data[:,-4:]                     # making the last four columns the features
y = data[:,5]                       # [:,n] - making the (n+1)th column the target

nfeat = len(X[0])

# shuffle the data
#shuffle_index = np.random.permutation(len(X))
#X, y = X[shuffle_index], y[shuffle_index]

# split Xdata and ydata in train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=9)

#PCA
print ('PCA logs')
pca = PCA(n_components=nfeat )
pca.fit(Xtrain)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
Xtrain =  pca.transform(Xtrain)
Xtest =  pca.transform(Xtest)

# MLP regressor
# play with layers structure here
# ---
nlayers = 1                                                      # number hidden of layers
neurons = [400]                                                  # neurons per layer
#neurons = neurons + neurons[:len(neurons)-1][::-1]
#neurons = list (range(2,301))
layers = tuple ( neurons*nlayers )                              # FINAL structure of the NN 

# MLP regressor
#reg = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes = (400,10), verbose = False, random_state=9)
reg = MLPRegressor(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes = layers, max_iter = 1000, verbose = False, random_state=9)
# ---
# useful notes about MLP hyperparameters 
# solver: {‘lbfgs’, ‘sgd’, ‘adam’} <> Generally: adam (default) is the best, sgd is faster and lbfgs is too slow 
# activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’


# others regressors
#reg = RandomForestRegressor(max_depth=2, random_state=9)

# uncomment if wanna see all the parameters of the model
print(reg)

# fit model with trin data
model = reg.fit(Xtrain, ytrain)

# prediciton for test set
preds = model.predict(Xtest)
              
# sklearn regression scores
print('r2 (Pearson) score: ', r2_score(ytest,preds))
print('explained_variance_score: ', explained_variance_score(ytest,preds))
print('mean_absolute_error ', mean_absolute_error(ytest,preds))
print('mean_squared_error ', mean_squared_error(ytest,preds))
#print('mean_squared_log_error ', mean_squared_log_error(ytest,preds))
print('median_absolute_error ', median_absolute_error(ytest,preds))

# visualization of predections vs target
# ---
hm = len(ytest)
ytestsortind  = sorted(range(len(ytest)),key=lambda x:ytest[x])
ytestsort = ytest[ytestsortind[:hm]]
predssort = preds[ytestsortind[:hm]]

plt.title('Ordered test set target vs regression')
plt.xlabel('label')
plt.ylabel('M_crit200')
plt.plot(predssort, 'r-', label = 'predict')
plt.plot(ytestsort, 'k.', label = 'target')
plt.legend()
plt.xlim((78000,79000))
#plt.ylim((0,0.0001))
plt.show()
# ---

print ('job duration in s: ', datetime.now() - start)
