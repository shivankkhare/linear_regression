# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:29:47 2019

@author: SHIVANK
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('50_Startups.csv')
x = df.iloc[:,:-1].values
y = df.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

x  = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
check = regressor.predict([[0,0,100672,91790,249744]])
print(check)

plt.plot(x_test,y_test,color ='blue')
plt.plot(x_test,y_pred,color='red')