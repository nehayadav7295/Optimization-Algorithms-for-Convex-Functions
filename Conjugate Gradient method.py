#Conjugate Gradient Method of optimization

import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()


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

end = time.time()


print(f"runtime of the conjugate method is {end-start}")

        
    
