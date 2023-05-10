# EX - 05 Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries: numpy, matplotlib.pyplot, and scipy.optimize.
2. Load the data from the file "ex2data1.txt" using numpy.loadtxt and split it into input features X and target variable Y.
3. Visualize the data by creating a scatter plot with two classes: "Admitted" and "Not admitted".
4. Define the sigmoid function that maps any real number to the range [0, 1].
5. Plot the sigmoid function using a range of values.
6. Define the cost function for logistic regression and its gradient.
7. Add a column of ones to the input features to account for the intercept term.
8. Initialize the parameters theta with zeros.
9. Compute the initial cost and gradient using the initial parameters.
10. Print the initial cost and gradient.
11. Use the scipy.optimize.minimize function to minimize the cost function and find the optimal parameters.
12. Print the final cost and optimized parameters.
13. Define a function to plot the decision boundary by creating a grid of points and computing the corresponding predicted class.
14. Plot the decision boundary along with the data points.
15. Calculate the probability of admission for a student with exam scores [45, 85].
16. Define a function to predict the class labels based on the learned parameters.
17. Calculate the accuracy of the model by comparing the predicted labels with the actual labels.
18. Print the accuracy of the model## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Neha.MA
RegisterNumber:  212220040100
*/

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1.txt",delimiter=",")

X = data[:, [0,1]]
Y = data[:,2]

X[:5]

Y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10, 10, 100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J, grad

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
J, grad = costFunction(theta, X_train, y)
print(J)  
print(grad)  

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, X_train, y)
print(J) 
print(grad)  

def cost(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = - (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J


def gradient(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, h - y) / X.shape[0]
    return grad


X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),
                        method='Newton-CG', jac=gradient)
print(res.fun) 
print(res.x)  


def plotDecisionBoundary(theta, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot = np.hstack((np.ones((X_plot.shape[0], 1)), X_plot))
    y_plot = np.dot(X_plot, theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Not admitted")
    plt.contour(xx, yy, y_plot, levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x, X, y)


prob = sigmoid(np.dot(np.array([1, 45, 85]), res.x))
print(prob)  


def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    prob = sigmoid(np.dot(X_train, theta))
    return (prob >= 0.5).astype(int)


np.mean(predict(res.x, X) == y)
```

## Output:
<img width="668" alt="Screenshot 2023-05-08 at 3 47 41 PM" src="https://user-images.githubusercontent.com/71516398/236802362-b7b7cc0e-7d01-4a79-97f1-f6c32868a1ce.png">
<img width="668" alt="Screenshot 2023-05-08 at 3 47 56 PM" src="https://user-images.githubusercontent.com/71516398/236802366-8bfe862a-739e-47f0-bd78-66cdca88f670.png">
<img width="653" alt="Screenshot 2023-05-08 at 3 48 08 PM" src="https://user-images.githubusercontent.com/71516398/236802371-9226619b-6d9e-4050-8823-9fc260645dbd.png">
<img width="455" alt="Screenshot 2023-05-08 at 3 49 49 PM" src="https://user-images.githubusercontent.com/71516398/236802380-8caee2d2-e62e-4b1c-a2a0-aaae45f58bba.png">
<img width="455" alt="Screenshot 2023-05-08 at 3 49 54 PM" src="https://user-images.githubusercontent.com/71516398/236802385-f7f05c6e-ffda-4f90-b102-8a9851cde847.png">
<img width="455" alt="Screenshot 2023-05-08 at 3 50 01 PM" src="https://user-images.githubusercontent.com/71516398/236802388-03722d86-060e-4bdb-83bb-d08dfafc1433.png">
<img width="642" alt="Screenshot 2023-05-08 at 3 50 19 PM" src="https://user-images.githubusercontent.com/71516398/236802391-d4dfdb4c-eb11-4fe3-9f7b-d55f7426a4f1.png">
<img width="642" alt="Screenshot 2023-05-08 at 3 50 27 PM" src="https://user-images.githubusercontent.com/71516398/236802402-4e8f36e0-c76d-4433-8988-a47335e54b18.png">
<img width="642" alt="Screenshot 2023-05-08 at 3 50 32 PM" src="https://user-images.githubusercontent.com/71516398/236802406-077b01f4-cf57-4b5b-b165-1239ea2620d2.png">


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
