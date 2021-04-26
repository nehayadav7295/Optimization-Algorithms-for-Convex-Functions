# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:05:58 2021

@author: Aishwarya Mathur
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import random
import timeit
 
class Gradient_Optimization:
    
    def __init__(self,X,y):
        self.X=X
        self.y=y
      
    def cost_function(self,X,y,theta):
        """
        X: slope
        y: intercept
        theta: theta value 
        returns: cost function
        """
        m = len(y)
        predict = X.dot(theta)
        cost = (1/2*m) * np.sum(np.square(predict-y))
        return cost
     
    
    def gradient_descent(self,X, y, theta, tol=0.01, iteration=1000):
        """
        X: randomly generated value
        Y: Function of X, X^2
        theta: theta value
        tol: Tolerance value
        iteration: Number of iterations
        returns: theta value and cost history
        """
        
        cost_history = [0] * (iteration+1)
        cost_history[0] = self.cost_function(X, y, theta)# you may want to save initial cost
        step_size=1
        iters=0
        while step_size>tol and iters<iteration:
            for iteration in range(iteration):
                h = X.dot(theta)
                loss = h - y.ravel()
                gradient = X.T.dot(loss)/(2 * len(y))
                theta = theta - tol * gradient
                cost = self.cost_function(X, y, theta)
                cost_history[iteration+1] = cost
                iters=iters+1
        return theta, cost_history
    
     
    
    def coordinate_descent(self,X, y, theta, tol=0.01, iteration=1000):
        """
        X: randomly generated value
        Y: Function of X, X^2
        theta: theta value
        tol: Tolerance value
        iteration: Number of iterations
        returns: theta value and cost history
        """
        cost_history = [0] *(iteration+1)
        cost_history[0] = self.cost_function(X, y, theta)
        step_size=1
        iters=0
        while step_size>tol and iters<iteration:
            for iteration in range(iteration):
                for i in range(len(theta)):
                    dele = np.dot(np.delete(X, i, axis=1), np.delete(theta, i, axis=0))
                    theta[i] = np.dot(X[:,i].T, (y.ravel() - dele))/np.sum(np.square(X[:,i]))
                    cost = self.cost_function(X, y, theta)
                    cost_history[iteration+1] = cost
                    iters=iters+1
        return theta, cost_history
    
    
    
    def batch_gradient_descent(self,X,y,theta,tol=0.01,iterations=1000,batch_size =20):

        """
        X: randomly generated value
        Y: Function of X, X^2
        theta: theta value
        tol: Tolerance value
        iteration: Number of iterations
        returns: theta value and cost history
        """ 
        m = len(y)
        c=0
        cost_history = np.zeros(iterations)
        n_batches = int(m/batch_size)
        learning_rate=0.01
        for it in range(iterations):
            cost =0.0
            indices = np.random.permutation(m)
            X = X[indices]
            y = y[indices]
            for i in range(0,m,batch_size):
                X_i = X[i:i+batch_size]
                y_i = y[i:i+batch_size]
                X_i = np.c_[np.ones(len(X_i)),X_i]
                prediction = np.dot(X_i,theta)
                theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))
                cost += self.cost_function(theta,X_i,y_i)
                c=c+it
            if c%20 ==  0:   
              cost_history[it]  = cost    
        cost_history=np.ma.masked_equal(cost_history,0)    
        cost_history.compressed()  
        return theta, cost_history
    
    def time_function1(self,X,y,theta,tol,iteration):
        start_time = timeit.default_timer()
        self.gradient_descent(X, y, theta, tol=0.01, iteration=1000)
        print("Time taken for Gradient Descent:",timeit.default_timer() - start_time)
        start_time = timeit.default_timer()
        self.coordinate_descent(X,y,theta,tol=0.01,iteration=1000)
        print("Time taken for Coordinate Descent:",timeit.default_timer() - start_time)
        
    
    def time_function2(self,X,y,theta,tol,iteration):
        start_time = timeit.default_timer()
        self.batch_gradient_descent(X, y, theta, 0.01, 1000)
        print("Time taken for Batch Gradient Descent:",timeit.default_timer() - start_time)
        
    
    def plot(self,gd_ret,gd_xret,cd_ret,cd_xret,bgd_ret,bgd_xret):
        iteration=1000
        plt.plot(range(len(gd_xret)), gd_xret, label="GradientDescent")
        plt.plot(range(len(cd_xret)), cd_xret, label="CoordinateDescent")
        plt.plot(range(iteration), bgd_xret, label="BatchGradientDescent")
        plt.ylabel('Theta')
        plt.xlabel('Minimum values of Cost Function')
        plt.legend()
        plt.show()
        

if __name__ == "__main__":
    print("----COMPARATIVE STUDY OF ALGORITHMS TO OPTIMIZE COVEX FUNCTIONS----")
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


    
    
    

    
    