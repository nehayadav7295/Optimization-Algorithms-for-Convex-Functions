# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:07:01 2021

@author: HP PAV -15 AU111TX
"""

import matplotlib.pyplot as plt
import matplotlib


def gd(f=(lambda x:x**2+3),tol=0.00001,iterations=1000):
    """
    Parameter f: gradient of our function
    tol: tolerace (tells us when to stop the algo
    iterations: max no. of iterations

    """
    points=[]
    current_x=-1  #the algo starts at this point (lower bound)
    rate=0.01  #learning rate
    iters=0    # iteration counter
    step_size=1
    while step_size>tol and iters<iterations:
        previous_x=current_x
        current_x= current_x-rate*f(previous_x) #gradient descent
        step_size=abs(current_x-previous_x) #change in x
        iters=iters+1 #iteration count
        points.append(current_x)
        print("Iterations",iters,"\nx value is",current_x)
    print("Number of iterations",iterations)
    print("Number of iterations required to reach tolerance",len(points))
    print("The local minimum occurs at",current_x)   
    
    plt.plot(points,'r.')
    plt.xlabel('Number of iterations')
    plt.ylabel('Value of x')
    plt.show()


gd()