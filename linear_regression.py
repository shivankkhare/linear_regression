# -*- coding: utf-8 -*-
"""
Created on Sun May  5 00:55:43 2019

@author: SHIVANK
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('employee_data.csv')

x = df.iloc[:,3:4].values
y = df.iloc[:,-1].values


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size =1/3 , random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
plt.plot(x_test,y_test,color ='blue')
plt.plot(x_test,y_pred,color='red')
plt.title('salary vs age understanding')
plt.xlabel('age')
plt.ylabel('salary')