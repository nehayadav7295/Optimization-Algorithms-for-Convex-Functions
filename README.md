# Optimization-Algorithms-for-Convex-Functions
The Package's function is to optimize convex functions using Interval bisection Method,
Golden Section Search, Particle swarm optimization, Gradient Descent, batch Gradient
Descent, Conjugate Descent and Coordinate Descent

The package consist of following files:

1. _init_.py : Initialises the Optimization class and run some test inputs
2. Python_project.py : It contains a Optimization class optimizes a function 
    using Interval bisection Method Golden Section Search, Particle swarm optimization, 
    Gradient Descent and batch Gradient Descent
    
    Graph is plotted as per each iteration taken by each algorithm to reach tolerance or 
    minima of the function
    
    Time taken for each algorithm to optimize for given constraints is displayed
    
    Second graph indicates the number of iterations and the current position of x at that 
    iteration for each algorithm
     
    Sample Inputs : 
    
    x = Optimization((x-3)**2+3, Upper_bound=15, Lower_bound=-14, Max_iter=100)
    print(x.plot())
    x=sym.Symbol('x')
    x1 = Optimization(x**2+x+3, Upper_bound=5, Lower_bound=-14, Max_iter=100)
    print(x1.plot())
    x=sym.Symbol('x')
    x1 = Optimization(x**3+x+3, Upper_bound=10, Lower_bound=-14, Max_iter=100)
    print(x1.plot())
 
    
