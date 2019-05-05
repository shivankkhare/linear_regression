# -*- coding: utf-8 -*-
"""
Created on Mon May  6 00:06:21 2019

@author: SHIVANK
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')

x = df.iloc[:,1:2].values
y = df.iloc[:,2].values
#we will not divide dataset into test set and train set

# linear regression model just to check an compare it by polynomial model
from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(x,y)

#polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

#PLOTTING PLOTS
plt.scatter(x,y)
plt.plot(x,Lin_reg.predict(x),color = 'red')
plt.scatter(x,y)
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)))
plt.scatter(2.5,lin_reg_2.predict([[1,2.5,6.25,15.625,39.0625]]))
