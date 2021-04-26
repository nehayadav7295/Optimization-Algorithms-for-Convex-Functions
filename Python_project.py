
import sys
import math
import random

import numpy as np

import matplotlib.pyplot as plt
import sympy as sym

import timeit

class Optimization:
    
    def __init__(self, func, Upper_bound=5, Lower_bound=-14, Max_iter=100):
        
        if self.Convexity_check(func,Lower_bound, Upper_bound) is True:
            print("Convex Function is",func)
            self.gdfunc=func
            x=sym.Symbol('x')
            self.func=sym.lambdify(x,func)
            
            #print("FUnction is",func)
        else:
            print(" The input function is not Convex function")
            sys.exit()
        self.Upper_bound=Upper_bound
        self.Lower_bound=Lower_bound
        self.Max_iter=Max_iter 
        
    def Func(self, x):
        """
        

        Parameters
        ----------
        x : float/int
            Value of X passed to function 
            

        Returns
        -------
        y : float/int
            value of function at x provided in input

        """
        y=self.func(x)
        return y     

    def I_bisection(self,a,b,tol,itr):
        """
        Optimise function self.Func using Interval Bisection Search in the given upper
        limitx and lower limits with respect to given tolerance value

        Parameters
        ----------
        a : float/int
            Lower bound for the function.
        b : float/int
            Upper bound for the function.
        tol : float
            Tolerance value for algorithm.
        itr : int
            Maximum number of iterations.

        Returns
        -------
        ibs : List
            [[x value of the iterations steps],value of x at minimum point, number of iterations required to reach minima].

        """
      
        ibs=[]
        iterations=0
        ibs_x=[]
        while iterations < itr:
          
          if (a+b)/2 > tol:
             a=a
             b=(a+b)/2
             ibs_x.append(b)
             #print("Upper Bound = ",b)
             
             
          elif (a+b)/2 <  tol :
             a=(a+b)/2
             b=b
             #print("Lowe Bound =",a)
             ibs_x.append(a)
             
             
          elif (a+b)/2 == tol:
             ibs_x.append((a+b)/2)
             #print(" Sol =" , (a+b)/2)
          iterations=iterations+1
             
             
        #print("IBS =",self.Func((a+b)/2)," iter=",iterations) 
        ibs.append(ibs_x)
        ibs.append((a+b)/2)
        ibs.append(iterations)
        
        return ibs
    
    def gss(self,LB,UB,tol,itr):
       """
        Optimise function self.Func using Golden Section Search in the given upper
        limit and lower limits with respect to given tolerance value

        Parameters
        ----------
        LB : float/int
            Lower bound for the function.
        UB : float/int
            Upper bound for the function.
        tol : float
            Tolerance value for algorithm.
        itr : TYPE
            Maximum number of iterations.

        Returns
        -------
        gss : List
            [[x value of the iterations steps],value of x at minimum point, number of iterations required to reach minima].

        """
           
       GoldenRatio = (math.sqrt(5) + 1) / 2
       
       iterations=0
       gss=[]
       gss_x=[LB,UB]
       
       c = UB - (UB - LB) / GoldenRatio
       d = LB + (UB - LB) / GoldenRatio
       while abs(UB - LB) > tol and iterations < itr:
           if self.Func(c) < self.Func(d):
               UB = d
               gss_x.append(UB)
               iterations+=1
           else:
               LB = c
               
               gss_x.append(LB)
               iterations+=1
           c = UB - (UB - LB) / GoldenRatio
           d = LB + (UB - LB) / GoldenRatio
      
        
       #print(" best at %.15f"% ((UB + LB)/2) , "itr = ",iterations)
       gss.append(gss_x)
       gss.append((LB+UB)/2)
       gss.append(iterations)
       
       return gss

    
    def gradient_descent(self, X ,eta, tol,iter):
        """
        Optimise function self.Func using Gradient Descent using the initial value 
        and tolerance value. Gradient used to move to next step is calculated using
        differential function of the main function

        Parameters
        ----------
        X : float/int
            Starting point.
        eta : float
            Learning rate for gradient descent.
        tol : float
            Tolerance value for algorithm.
        iter : TYPE
            Maximum number of iterations.

        Returns
        -------
        gd : List
            [[x value of the iterations steps],value of x at minimum point, number of iterations required to reach minima].

        """
        gd=[]
        gd_x=[X]
        iteration=0
        # current_pt=X
        first_derivative=sym.diff(self.gdfunc)
        #print(first_derivative)
        x=sym.Symbol('x')
        first_derivative=sym.lambdify(x,first_derivative)
        learn_rate=eta
    
    
        prev_x=X
        new_x=prev_x -(learn_rate*first_derivative(prev_x))
        gd_x.append(new_x)
        #print("prev_x = ",prev_x," Next x = ",new_x)
        for i in range(iter):
            prev_x=new_x
            #print(prev_x)
            new_x=prev_x -(learn_rate*first_derivative(prev_x))
            gd_x.append(new_x)
           # print("x = ",new_x,"Gradient =",learn_rate*self.func(prev_x))
            if abs(self.func(new_x)) <= self.func(tol) :
                break
            iteration=iteration+1
        #print("Best at GD x= ",new_x)
        gd.append(gd_x)
        gd.append(new_x)
        gd.append(iteration)

        return gd
    
    def b_gradient_descent(self, LB,UB,eta, tol,iter):
        """
        Optimise function self.Func using Batch Gradient Descent. The gradient is 
        calculated using data set in between Lower and Upper bound.
        The main value iterates only once using final gradient calculated using
        data set.

        Parameters
        ----------
        LB : float/int
            Lower bound for the function.
        UB : float/int
            Upper bound for the function.
        eta : float
            Learning rate for gradient descent.
        tol : float
            Tolerance value for algorithm.
        iter : TYPE
            Maximum number of iterations.

        Returns
        -------
        bgd : List
            [[x value of the iterations steps],value of x at minimum point, number of iterations required to reach minima].


        """
        bgd=[]
        bgd_x=[LB]
        iteration=0
        # current_pt=X
        first_derivative=sym.diff(self.gdfunc)
        #print(first_derivative)
        x=sym.Symbol('x')
        first_derivative=sym.lambdify(x,first_derivative)
        learn_rate=eta
        
        new_x=LB
        bgd_x.append(LB)
        
        for i in range(iter):
            for j in np.arange(LB,UB,0.1):
               prev_x=new_x
               new_x=prev_x-(learn_rate*first_derivative(prev_x))
               #print("i = ",j,"gradient =",(learn_rate*first_derivative(j)),iteration)
               iteration=iteration+1
               #print(iteration)
               if iteration >=iter:
                 break  
            if new_x <= tol:
               #print("new_x = ",new_x,"gradient =",(learn_rate*first_derivative(prev_x)), iteration)  
               break
        
           
          
        
        #print(new_x)
        bgd_x.append(new_x)
        
        
        bgd.append(bgd_x)
        bgd.append(new_x)
        bgd.append(iteration)

        return bgd


    
    def particle_Swarm(self, func, initial, bound, particles, max_iter):
        w = 0.72984
        c1 = 2.05
        c2 = 2.05
        target = 0.001
        
      
        target_error = 0.001
        
        pso=[]
        x_pso=[initial]
        x_pso.append(bound)
        
       
        particle_position = np.array([[initial],[bound]])
        
        pbest_position = particle_position
        pbest = np.array([float('inf') for _ in range(particles)])
        gbest = float('inf')
        gbest_position = np.array([float('inf'), float('inf')])
        
        velocity_vector = ([np.array([0]) for _ in range(particles)])
        iteration = 0
        while iteration < max_iter:
            for i in range(particles):
                fitness_cadidate = self.Func(particle_position[i])
                #print(fitness_cadidate, ' ppv = ', particle_position_vector[i])
                
                if(pbest[i] > fitness_cadidate):
                    pbest[i] = fitness_cadidate
                    pbest_position[i] = particle_position[i]
        
                if(gbest > fitness_cadidate):
                    gbest_fitness_value = fitness_cadidate
                    gbest_position = particle_position[i]
        
            if(abs(gbest_fitness_value - target) < target_error):
                break
                
            for i in range(particles):
                r1=c1*random.uniform(0,0.5)
                r2=c2*random.uniform(0,0.5)
                new_velocity = (w*velocity_vector[i]) + (r1) * (pbest_position[i] - particle_position[i]) + (r2) * (gbest_position-particle_position[i])
                #print("Velocity",new_velocity ," r1=",r1," r2 = ",r2)
                new_position = new_velocity + particle_position[i]
                x_pso.append(new_position[0])
                #print("New position = ",particle_position_vector[i])
                particle_position[i] = new_position
                w = (0.4/max_iter**2) * (iteration - max_iter) ** 2 + 0.4
                c1 = -3 * iteration / max_iter + 3.5
                c2 =  3 * iteration / max_iter + 0.5
            iteration = iteration + 1
        pso.append(x_pso)
        pso.append(gbest_position)  
        pso.append(iteration)
        #print("The best position is ", self.Func(gbest_position), "in iteration number ", iteration)
        return pso
    
    def plot(self):
        """
        This function plots graphs to compare performance of different algorithms
        Plots the graphs comparing following algothms:
        Interval Bisection Search vs Golden Section Search vs Particle swarm
        Gradient Descent vs Batch Gradient Descent 
        Returns
        -------
        None.

        """
        
        
        x_ibs=[] 
        x_gss=[]
        y_ibs=[] 
        y_gss=[]
        x_pso=[]
        x_bgd=[]
        y_bgd=[]
        y_pso=[]
        x_gd=[]
        y_gd=[]
        
        i=0.0000001
        
        # for k in range(1,51):
        #  i= random.uniform(0.00000001, 1)
        #  t_avg_ibs=[]
        #  t_avg_gss=[]
        #  for j in range(1,51):
          #L=random.randint(-100, 0)
          #U=random.randint(0, 100)
        max_iter=self.Max_iter  
        L=self.Lower_bound
        U=self.Upper_bound
        
        minima=self.gss(L,U,i,1000)
        #print("minima at X = ",minima[1])
        x_ibs.append(self.I_bisection(L,U,minima[1],max_iter)[0])
        x_gss.append(self.gss(L,U,i,max_iter)[0])
        x_pso.append(self.particle_Swarm(self.func, L, U, 2, max_iter)[0])
        x_gd.append(self.gradient_descent(X=U ,eta=0.01, tol=minima[1],iter= max_iter)[0])
        x_bgd.append(self.b_gradient_descent(LB=L,UB=U ,eta=0.01, tol=minima[1],iter=max_iter)[0])
        #print(x_pso)
        for i in x_ibs[0]:
          #print(self.Func(i))  
          y_ibs.append(self.Func(i))
        for i in x_gss[0]:
          y_gss.append(self.Func(i))          
        for i in x_pso[0]:
          y_pso.append(self.Func(i))  
        for i in x_gd[0]:
          y_gd.append(self.Func(i)) 
        for i in x_bgd[0]:
          y_bgd.append(self.Func(i))             
        #print(y_gss)

        plt.plot(x_ibs[0], y_ibs, 'r.')
        plt.plot(x_gss[0], y_gss, '.')
        plt.plot(x_pso[0], y_pso, 'y.')
        #plt.plot(x_gd[0], y_gd, 'y.')
        #plt.plot(x_bgd[0], y_bgd, 'k.')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.suptitle('Interval Bisection Search (Red) vs Golden Section Search (Blue) vs Particle swarm optimization (Green)')
        #plt.axis([0, 100, 0.00000001, 1])   
        plt.show()
        plt.plot(x_gd[0], y_gd, 'r.')
        plt.plot(x_bgd[0], y_bgd, 'k.')
        plt.xlabel('x')
        plt.ylabel('y')        
        plt.suptitle('Gradient Descent (Red) vs Batch Gradient Descent (Black) ')
        
        plt.show()
        
        start_time = timeit.default_timer()
        ibs=self.I_bisection(L,U,minima[1],max_iter)
        print(" Execution time for Interval bisection Method is", timeit.default_timer() - start_time,"s")
        start_time = timeit.default_timer()
        gss=self.gss(L,U,i,max_iter)
        print(" Execution time for Golden Section Search is", timeit.default_timer() - start_time,"s")
        start_time = timeit.default_timer()
        pso=self.particle_Swarm(self.func, L, U, 2, max_iter)
        print(" Execution time for Particle swarm optimization is", timeit.default_timer() - start_time,"s")
        start_time = timeit.default_timer()
        gd=self.gradient_descent(X=U ,eta=0.01, tol=minima[1],iter= max_iter)
        print(" Execution time for Gradient Descent is", timeit.default_timer() - start_time,"s")
        start_time = timeit.default_timer()
        bgd=self.b_gradient_descent(LB=L,UB=U ,eta=0.01, tol=minima[1],iter=max_iter)
        print(" Execution time for Batch Gradient Descent is", timeit.default_timer() - start_time,"s")
        plt.plot(ibs[1], ibs[2], 'r.')
        plt.text(ibs[1], ibs[2],"IB")
        plt.plot(gss[1], gss[2], '.')
        plt.text(gss[1], gss[2]," GSS")
        plt.plot(pso[1], pso[2], 'y.')
        plt.text(pso[1], pso[2],"    PSO")
        plt.plot(gd[1], gd[2], 'g.')
        plt.text(gd[1], gd[2],"          GD ")
        plt.plot(bgd[1],bgd[2], 'k.')
        plt.text(bgd[1], bgd[2],"       Batch_GD")
        
        plt.xlabel('Value of X')
        plt.ylabel('NUmber of iteration')        
        plt.suptitle('Number of iterations vs minimum value of x')
        
        plt.show()
       
     

    
    def Convexity_check(self,func,a,b):
        
        x=sym.Symbol('x')
        first_derivative=sym.diff(func)
        second_derivative=sym.diff(first_derivative)
        #print(second_derivative)
        second_derivative=sym.lambdify(x,second_derivative)
        #print(first_derivative,a,b)
        
        
        #Checking convexity between a ans b
        convex=True
        for i in range(a,b):
            if second_derivative(i) < 0:
                
                convex=False
            elif second_derivative(i) >=0:
                 pass
       # print(convex)   
        return convex      
    

                   


if __name__ == "__main__":
    print("----COMPARATIVE STUDY OF ALGORITHMS TO OPTIMIZE COVEX FUNCTIONS----")
    x=sym.Symbol('x')
    #x = Optimization(x**2+x+3)
    x = Optimization((x-3)**2+3, Upper_bound=15, Lower_bound=-14, Max_iter=100)
    print(x.plot())
    
    x=sym.Symbol('x')
    x1 = Optimization(x**3+x+3, Upper_bound=10, Lower_bound=-14, Max_iter=100)
    print(x1.plot())
    
    x=sym.Symbol('x')
    x1 = Optimization(x**2+x+3, Upper_bound=5, Lower_bound=-14, Max_iter=100)
    print(x1.plot())
  
