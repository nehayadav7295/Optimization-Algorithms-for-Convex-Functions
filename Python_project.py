

import math
import random
import scipy.optimize
import numpy 
import matplotlib.pyplot as plt
import sympy

class Optimization:
    
    def __init__(self, func=(lambda x: 2*x), Upper_bound=100, Lower_bound=0, Max_iter=1000):
        self.func=func
        self.Upper_bound=Upper_bound
        self.Lower_bound=Lower_bound
    
    def Func(self, x):
        y=lambda x: x**2
        return y     

    def I_bisection(self, f,a,b,tol):
        """
        Optimise function f using Interval Bisection Search in the given upper
        limit and lower limits with respect to given tolerance value
        param f: Function of x, which is to be optimised
        param a: Lower bound for the function
        param b: Upper bound for the function
        param tol : Tolerance value for algorithm
        ## 
        returns [[x value of the iterations steps],[number of iterations],[Number of iterations required to reach the tolerance],[ x at minimum point]]
        
        returns: A list of [Number of iterations required to reach the tolerance, x at minimum point]
           """
        ibs=[]
        iterations=0
        ibs_x=[]
        while (b-a) > tol and iterations<1000:
        #print("a = ",a," b =",b)
          if f((a+b)/2) > 0:
             a=a
             b=(a+b)/2
             #ibs_x.append()
          else:
             a=(a+b)/2
             b=b
          iterations+=1
         
        ibs.append(iterations)
        ibs.append((a+b)/2)
        return ibs
    
    def gss(self,f,a,b,tol):
       """
        Optimise function f using Interval Bisection Search in the given upper
        limit and lower limits with respect to given tolerance value
        param f: Function of x, which is to be optimised
        param a: Lower bound for the function
        param b: Upper bound for the function
        param tol : Tolerance value for algorithm
        returns: A list of [Number of iterations required to reach the tolerance, x at minimum point]
           """        
       gr = (math.sqrt(5) + 1) / 2
       iterations=0
       gss=[]
       c = b - (b - a) / gr
       d = a + (b - a) / gr
       while (b - a) > tol and iterations<1000:
         
           if f(c) < f(d):
               b = d
           else:
               a = c
           iterations+=1
           # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
           c = b - (b - a) / gr
           d = a + (b - a) / gr
       gss.append(iterations)
       gss.append((a+b)/2)
       return gss


    def coordinate_descent(self, X, y, param,iter):
        pass
    
    def gradient_descent(self, X, y, par,eta,iter):
        pass
    
    
    def particle_Swarm(self, func, initial, bound, particles, max_iter):
        pass
    
    def plot(self):
        x_ibs=[] 
        x_gss=[]
        y_ibs=[] 
        y_gss=[]
        
        i=0.0000001
        for k in range(1,51):
         i= random.uniform(0.00000001, 1)
         t_avg_ibs=[]
         t_avg_gss=[]
         for j in range(1,51):
          L=random.randint(-100, 0)
          U=random.randint(0, 100)
          z=self.I_bisection(self.func,L,U,i)
          t_avg_ibs.append(z[0])
          t_avg_gss.append(self.gss(self.func,L,U,i)[0])
         x_ibs.append(math.log10(sum(t_avg_ibs)/len(t_avg_ibs)))
         y_ibs.append(i)
         #print("IBS x = ",sum(t_avg_ibs)/len(t_avg_ibs)," y = ", i)
         x_gss.append(math.log10(sum(t_avg_gss)/len(t_avg_gss)))
         #print("GSS x = ",sum(t_avg_gss)/len(t_avg_gss)," y = ", i)
         y_gss.append(i)
       
        plt.plot(x_ibs, y_ibs, 'r.')
        plt.plot(x_gss, y_gss, '.')
        plt.xlabel('log10 (Average Iterations)')
        plt.ylabel('$Tol')
        plt.suptitle('Interval Bisection Search (Red) vs Golden Section Search (Blue)')
        #plt.axis([0, 100, 0.00000001, 1])   
        plt.show()
     

    
    def Convexity_check(func,a,b):
        pass
    

                   


if __name__ == "__main__":
    print("----COMPARATIVE STUDY OF ALGORITHMS TO OPTIMIZE COVEX FUNCTIONS----")
    x = Optimization(lambda x: 2*x)
    print(x.plot())
    
#---------------------------------------------Optimization using Batch Gradient Method----------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt  #importing matplot for plotting
import numpy as np            #counts the number of observations per category
from scipy import stats 
from sklearn.datasets.samples_generator import make_regression

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

#Batch Gradient method function
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

#-----------------------------------------------Optimization using Conjugate Gradient method----------------------------------------------------------------------------------------------------------------   

#Conjugate Gradient Method of optimization

import matplotlib.pyplot as plt
import numpy as np


#creating a plot function
def plot_iterations(f, iterations, delta=0.1, box=1):
  
    a = np.arange(-box, +box, delta)
    b = np.arange(-box, +box, delta)
    X, Y = np.meshgrid(a, b)
    
    #Creating a function mesh
    Z = np.array([[f(np.array([x, y])) for x in a] for y in b])
    
    #creating a contour plot
    fig = plt.figure(figsize=(10, 5))
    jx = fig.add_subplot(1, 2, 1)
    cd = jx.contour(X, Y, Z)
    
    
    # error plot
    jx.clabel(cd, inline=1, fontsize=10)
    jx.plot(iterations[:, 0], iterations[:, 1], color='blue', marker='x', label='opt')
    jx = fig.add_subplot(1, 2, 2)
    jx.plot(range(len(iterations)), [f(np.array(step)) for step in iterations], 'xb-')
    jx.grid('on')
    jx.set_xlabel('iteration')
    jx.set_xticks(range(len(iterations)))
    jx.set_ylabel('f(x)')
  
#specifying a callback function between iterations.Function will be called back after each iteration
def call_back(data_list):
    """
    Function: Calling back function between iterations
    param data_list
    return: the called back function
    """
    def callback_function(x):
        data_list.append(x)
    return callback_function

#specifying a function to minimize f(x) = 9x^2 + 7y^2

f1 = lambda x: x.T.dot(np.diag([9,7])).dot(x)

#first and second-order characterizations

#function for jacobian vector for the first partial derivative
def df1(x):
    """
    Function: first partial derivative
    param x
    """
    return (np.diag([9,7]) + np.diag([9,7]).T).dot(x)

#function for the hessian matrix of the second partial derivative

def df2(x):
    """
    Function: second partial derivative 
    param x
    """
    return np.diag([18,14])

#conjugate gradient method

x0= [-1,1]
cg_data=[np.array(x0)]
from scipy.optimize import fmin_cg
res = fmin_cg(f1,x0,fprime=df1, callback=call_back(cg_data))

#plotting
plot_iterations(f1, np.array(cg_data))

        
    
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

