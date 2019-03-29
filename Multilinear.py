# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 10:01:08 2018

@author: Manit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('50_Startups.csv')
data.head()
X=data.iloc[:,:-1].values
Y=data.iloc[:,4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]= labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features = [3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the dumy variable Trap
X=X[:,1:]

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

y_predict=regressor.predict(X_test)

plt.plot(X_test,y_predict)
plt.scatter(X_test,y_test,marker='o',color='red')
plt.show()

#Buuilding the optimal model using Backward Wlimination
import statsmodels.formula.api as sm
np.append(arr =X, values=np.ones((50,1)).astype(int),axis=1)
X=np.append(arr = np.ones((50,1)).astype(int),values = X,axis=1)
X_opt=X[:,[0,3]]
regressor_OLS= sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()