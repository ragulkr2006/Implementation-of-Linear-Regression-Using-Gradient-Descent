# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
/*
```
Program to implement the linear regression using gradient descent.
Developed by: RAGUL K R
RegisterNumber: 212224240123
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print("\n")
print(X1_Scaled)
print("\n")
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

output :
# profit prediction Graph:
![image](https://github.com/user-attachments/assets/8c5d0241-fcbf-4625-a168-ef50bf444485)
![image](https://github.com/user-attachments/assets/e35460e7-3fd9-422f-b200-f21ba749e4be)
# Compute cost value :
![image](https://github.com/user-attachments/assets/c08c7b4f-b421-4216-9bef-3d293ed7f753)
# h(x)Value :
![image](https://github.com/user-attachments/assets/55cff9bf-39c1-47ef-a92c-60c27771f54f)
# Cost function using Gradient Descent Graph :
![image](https://github.com/user-attachments/assets/abca4a73-1ef1-478d-9d2d-33dce3a48556)
# Profit for the Population 35,000 :
![image](https://github.com/user-attachments/assets/dd722223-ba67-4d9e-9003-0ece954e6f93)
# Profit for the Population 70,000 :
![image](https://github.com/user-attachments/assets/ac18ed21-a091-4d34-9ad7-5d637002e160)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
