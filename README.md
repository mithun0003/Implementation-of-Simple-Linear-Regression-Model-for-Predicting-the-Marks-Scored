# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the standard libraries

2.set variables for asssigning dataset values

import linear regression from sklearn

assign the points for representing in the graph

predict the regression for marks by using the representation of the graph

compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')

df.head()
df.tail()

x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y

# Splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/2,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

# Displaying predicted values
y_pred
y_test

plt.scatter(x_test,y_test,color = "yellow")
plt.plot(x_train,regressor.predict(x_train),color = "purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test),color='blue')
plt.title("Hours vs scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()

mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse = np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
DATA HEAD

![image](https://github.com/user-attachments/assets/fbbd530b-9b71-41a0-80ad-a3104775f95a)

DATA TAIL

![image](https://github.com/user-attachments/assets/78f2d0b7-9564-4518-85cc-d198e7b21bcc)

ARRAY VALUES OF X

![image](https://github.com/user-attachments/assets/9c0b057f-2bb9-4426-b990-461eda8174ab)

ARRAY VALUES OF Y

![image](https://github.com/user-attachments/assets/f2966e0e-246e-4843-8718-69f7f6f41afb)

VALUES OF Y PREDICTION

![image](https://github.com/user-attachments/assets/c217f9fb-5418-4052-b11f-3612cd8907a7)

ARRAY VALUES OF Y TEST

![image](https://github.com/user-attachments/assets/bfd53ea1-bee0-45e2-85ad-8eb127321a70)


TRAINING SET GRAPH

![image](https://github.com/user-attachments/assets/aff31565-b697-424d-809d-83f32352d18b)

TESTING SET GRAPH

![image](https://github.com/user-attachments/assets/3dcb23c2-5835-4ea4-ac77-9bd14562e58b)

VALUES OF MSE, MAE AND RMSE

![image](https://github.com/user-attachments/assets/b8a796d5-9327-4572-9500-51293abd55f6)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
