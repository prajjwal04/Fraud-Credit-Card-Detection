# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing Libraries
import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

#Loading Datasets
data=pd.read_csv('creditcard.csv')
print(data.columns)
print(data.shape)
print(data.describe())
data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)

#Plotting histogram of each parameter
data.hist(figsize = (20,20))
plt.show()

#Determine number of fraud cases in dataset
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(Fraud)))
print('Valid Cases: {}'.format(len(Valid)))

#Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8 , square = True)
plt.show()