
from Python_project import Optimization
from GD_CD_BGD import Gradient_Optimization
import sympy as sym
from sklearn.preprocessing import StandardScaler
import numpy as np

from numpy import random


if __name__ == "__main__":
    print("----COMPARATIVE STUDY OF ALGORITHMS TO OPTIMIZE COVEX FUNCTIONS----")
    x=sym.Symbol('x')
    
    """
    Optimization class optimizes a function using Interval bisection Method,
    Golden Section Search, Particle swarm optimization, Gradient Descent and
    batch Gradient Descent
    
    Graph is plotted as per each iteration taken by each algorithm to reach tolerance or 
    minima of the function
    
    Time taken for each algorithm to optimize for given constraints is displayed
    
    Second graph indicates the number of iterations and the current position of x at that 
    iteration for each algorithm
    """
    x = Optimization((x-3)**2+3, Upper_bound=15, Lower_bound=-14, Max_iter=100)
    print(x.plot())
    
    
    
    x=sym.Symbol('x')
    x1 = Optimization(x**2+x+3, Upper_bound=5, Lower_bound=-14, Max_iter=100)
    print(x1.plot())
    
    
    """
    Gradient Optimization class optimizes values of X,y using Gradient Descent, Coordinate Descent
    and Batch Gradient Descent
    
    Graph is plotted for all algorithms between the 'theta value and cost value'
    
    Time taken for each algorithm for optimizations is displayed

    """
    X = random.rand(100)
    y = X**2
    tol=0.1
    iteration=1000
    O = Gradient_Optimization(X,y)
    theta = np.random.randn(2,1)
    O.time_function2(X,y,theta,tol,iteration)
    bgd_ret,bgd_xret = O.batch_gradient_descent(X, y, theta)
    X=X.reshape(-1,1)
    y = y.reshape(-1,1)
    X = StandardScaler().fit_transform(X)  # for easy convergence
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    theta = np.zeros(X.shape[1])
    gd_ret, gd_xret = O.gradient_descent(X, y, theta)
    cd_ret, cd_xret = O.coordinate_descent(X, y, theta)
    O.time_function1(X,y,theta,tol,iteration)
    O.plot(gd_ret,gd_xret,cd_ret,cd_xret,bgd_ret,bgd_xret)
    

    
    """
    Note:
        
        the input function must be concave .
        If the value of Upper ans lower bound is high it will give an error for GD and Batch GD
        In that case please reduce the eta in the plot() function that is passed to the respective 
        functions to 0.0001, The float value gets very high therefore to reduce that eta is changed
        Also, After changing the eta the step size becomes significantly low therefore slow sprocessing
        
    
    """

    