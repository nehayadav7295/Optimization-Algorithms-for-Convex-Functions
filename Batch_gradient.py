import matplotlib.pyplot as plt  #importing matplot for plotting
import numpy as np            #counts the number of observations per category
from scipy import stats 
from sklearn.datasets.samples_generator import make_regression
import time

start = time.time()

#creating a simple dataset containing one feature and one target column which is continuos in nature(regression problem)
X, y = make_regression(n_samples = 10000, 
                       n_features=1, 
                       n_informative=1, 
                       noise=20,
                       random_state=2000)

#Step 1 (Finding the slope and the intercept) using linregress 
x = X.flatten()
slope, intercept,_,_,_ = stats.linregress(x,y)
print (slope)
print (intercept)

#Reach the slope and intercept using batch gradient descent method.

y = y.reshape(-1,1) #making it into a two dimensional array

#cost function
def cost_function(theta,X,y):
    """
    Function: Calculates the cost
    param theta: theta value
    X: slope
    y: intercept
    returns: cost function
    """
    m = len(y)
    predict = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predict-y))
    return cost


#Batch gradient method
"""


"""
def batch_gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    """
    Function to optimize the cost value using batch gradient descent method
    param X : Slope
    param Y : Intercept
    learning_rate : 0.01
    iterations : 100
    Returns: theta, cost_history, theta_history

    """
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
        theta_history[it, :] = theta.T
        cost_history[it] = cost_function(theta, X, y)
    return theta, cost_history, theta_history


#taking 1000 iterations and a learning rate of 0.05
lr = 0.05
n_iter = 1000
theta = np.random.randn(2, 1)
X_b = np.c_[np.ones((len(X), 1)), X]
theta, cost_history, theta_history = batch_gradient_descent(X_b, y, theta, lr, n_iter)
print("theta0: {:0.2f},\n theta1:{:0.2f}".format(theta[0][0], theta[1][0]))
print("Final Cost Value/MSE:  {:0.2f}".format(cost_history[-1]))

#plotting the graph
fig,ax = plt.subplots(figsize=(10,6))
ax.set_ylabel('Theta(J)')
ax.set_xlabel('Iterations')
_=ax.plot(range(n_iter),cost_history,'b.')

end = time.time()

print(f"running time of batch_gradient method is{end to start}")