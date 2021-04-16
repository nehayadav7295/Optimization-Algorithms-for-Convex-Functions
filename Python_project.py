

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
    
    def b_gradient_descent(self, alpha, x, y, numIterations):
        pass

    def conj_grad(self, x,A,b,c, min_iter, tol):
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
    


