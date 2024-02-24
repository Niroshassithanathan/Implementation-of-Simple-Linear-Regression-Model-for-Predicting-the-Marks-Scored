# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Import pandas, numpy and sklearn

2.Calculate the values for the training data set

3.Calculate the values for the test data set

4.Plot the graph for both the data sets and calculate for MAE, MSE and RMSE

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:NIROSHA S
RegisterNumber: 212222230097
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

##  splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred

## graph plot for training data
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

## graph plot for test data
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="purple")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE= ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
### df.head()
![image](https://github.com/Niroshassithanathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418437/dd7aa8ce-348b-4170-bafc-aed8d843210d)

### df.tail()
![image](https://github.com/Niroshassithanathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418437/c8d7a368-59df-4da5-9172-68c5668bc972)

### ARRAY VALUE OF X
![image](https://github.com/Niroshassithanathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418437/07cf60d3-7f79-489a-8f86-9d30bd0f5d5d)

### ARRAY VALUE OF Y
![image](https://github.com/Niroshassithanathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418437/b582a706-c64f-4c7d-a325-cfdaf9fa6cdc)

### VALUES OF Y PREDICTION
![image](https://github.com/Niroshassithanathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418437/7d58ab65-1640-4921-9b75-116b15d1cab8)

### ARRAY VALUES OF Y TEST
![image](https://github.com/Niroshassithanathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418437/4bf2a7f2-4d5c-41a5-ba28-917e3ea128f3)

### TRAINING SET GRAPH
![image](https://github.com/Niroshassithanathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418437/487a4e64-aa71-4e70-a5e6-892e05b43917)

### TEST SET GRAPH
![image](https://github.com/Niroshassithanathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418437/7f7cbccf-9826-4ccc-944b-4cc1b7077d4a)

### VALUES OF MSE,MAE AND RMSE
![image](https://github.com/Niroshassithanathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418437/18a0f653-2793-468a-aa7a-f64792dbf4cd)

![image](https://github.com/Niroshassithanathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418437/7be6710c-ade7-4ca0-820d-07942d8e3dd7)

![image](https://github.com/Niroshassithanathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418437/08782f52-09b5-49b2-bbee-a56579c8ac5c)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
