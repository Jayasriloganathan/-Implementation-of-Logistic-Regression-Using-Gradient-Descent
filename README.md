# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize parameters

2.Compute hypothesis with sigmoid

3.Compute cost (optional)

4.Update weights and bias using gradients

5.Make predictions 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Jayasri L
RegisterNumber:  
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Placement_Data.csv')
data
data=data.drop('sl_no',axis=1)
data=data.drop('salary',axis=1)
data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data.dtypes
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data

y = data['status'].values # Define y as the target variable
x = data.drop('status', axis=1).values # Define x as the feature set
theta = np.random.randn(x.shape[1])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta-=alpha * gradient
    return theta

theta=gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred = predict(theta,x)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy: ",accuracy)
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```
## Output:

<img width="1435" height="591" alt="image" src="https://github.com/user-attachments/assets/a180935d-4482-4520-a80b-34328316ed34" />

<img width="1210" height="785" alt="image" src="https://github.com/user-attachments/assets/4707f688-8cd9-42a3-bf82-914554695ef1" />

<img width="1172" height="692" alt="image" src="https://github.com/user-attachments/assets/5359d5b6-45ef-4024-a44b-5a23e04d4af8" />

<img width="1338" height="544" alt="image" src="https://github.com/user-attachments/assets/ece0c5ff-e79e-4965-8e27-cd01df371861" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

