# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:51:52 2018

@author: Admin
"""

#importing the libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing the dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values 

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap 
X=X[:, 1:]

#splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train ,y_test= train_test_split(X,y, test_size=0.2, random_state=0)

#feature scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)"""

#fitting multiple LR to train set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
 
#predicting the test set
y_pred=regressor.predict(X_test)
 
#build the optimum model using bakcward elimination
import statsmodels.formula.api as sm
"""adding one column for b0 which is constant in front as x0=1 """
X= np.append(arr= np.ones((50,1)).astype(int), values=X, axis=1)
"""step 2  and 3 of BE Model"""
X_opt= X[:, [0,1,2,3,4,5]]
regressor.OLS= sm.OLS(endog= y, exog= X_opt).fit()
regressor.OLS.summary()
"""step 4 and 5 (remove the index 2 with highest p)"""
X_opt= X[:, [0,1,3,4,5]]
regressor.OLS= sm.OLS(endog= y, exog= X_opt).fit()
regressor.OLS.summary()
"""repeating until we get FIN"""
X_opt= X[:, [0,3,4,5]]
regressor.OLS= sm.OLS(endog= y, exog= X_opt).fit()
regressor.OLS.summary()
X_opt= X[:, [0,3,5]]
regressor.OLS= sm.OLS(endog= y, exog= X_opt).fit()
regressor.OLS.summary() 
X_opt= X[:, [0,3]] 
regressor.OLS= sm.OLS(endog= y, exog= X_opt).fit()
regressor.OLS.summary()
 """....THUS R&D SPEND GIVES THE MAXIMUM PROFIT...."""

