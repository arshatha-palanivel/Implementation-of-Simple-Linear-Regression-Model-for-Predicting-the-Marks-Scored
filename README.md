# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```py
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: Arshatha 

RegisterNumber:  212222230012
```
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()

#segregating data to variables
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='yellow')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```


## Output:
![2 01](https://github.com/arshatha-palanivel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118682484/9800abca-14a5-49ec-b1e5-b18c3beb01ed)

![2 02](https://github.com/arshatha-palanivel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118682484/6f2469df-5e76-4ec6-b739-7808488ba522)

![2 03](https://github.com/arshatha-palanivel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118682484/798e9de1-f25e-49e8-867c-73abb01e1a09)

![2 04](https://github.com/arshatha-palanivel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118682484/b2eb7889-101a-4c77-8a91-99ed53046f15)

![2 05](https://github.com/arshatha-palanivel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118682484/f437bc50-69a1-4dbb-9937-5016368815d8)

![2 06](https://github.com/arshatha-palanivel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118682484/d32f53c2-9e17-483b-bb6c-fbb05aac3557)

![2 07](https://github.com/arshatha-palanivel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118682484/2b7d5d21-631b-43e9-a441-c034302f0c56)

![2 08](https://github.com/arshatha-palanivel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118682484/85a2f456-165b-4ab8-9372-3cf243416573)

![2 09](https://github.com/arshatha-palanivel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118682484/207a42ef-0741-409e-87ba-3a7ce71a4518)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
