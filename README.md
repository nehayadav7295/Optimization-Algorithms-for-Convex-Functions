# Optimization-Algorithms-for-Convex-Functions
The Package's function is to optimize convex functions using Interval bisection Method,
Golden Section Search, Particle swarm optimization, Gradient Descent, batch Gradient
Descent, Conjugate Descent and Coordinate Descent

The package consist of following files:

1. _init_.py : Initialises the Optimization class and run some test inputs
2. Python_project.py : It contains a Optimization class optimizes a function using Interval bisection Method Golden Section Search, Particle swarm optimization,         Gradient Descent and batch Gradient Descent Graph is plotted as per each iteration taken by each algorithm to reach tolerance or minima of the function
    Time taken for each algorithm to optimize for given constraints is displayed
    Second graph indicates the number of iterations and the current position of x at that iteration for each algorithm
     
    Sample Inputs : 
    
    x = Optimization((x-3)**2+3, Upper_bound=15, Lower_bound=-14, Max_iter=100)
    print(x.plot())
    x=sym.Symbol('x')
    x1 = Optimization(x**2+x+3, Upper_bound=5, Lower_bound=-14, Max_iter=100)
    print(x1.plot())
    x=sym.Symbol('x')
    x1 = Optimization(x**3+x+3, Upper_bound=10, Lower_bound=-14, Max_iter=100)
    print(x1.plot())
    
    
3. Batch_gradient.py : In batch gradient a simple data set is created with one feature and one target column which is continuous in nature(regression). This will be    the input.Slope and the intercept is found for the data using linregress from sklearn package. We reach the slope and the intercept value using batch gradient      descent method.

   The cost function which is the difference between the predicted and the original value is returned. Furthermore, the weights are adjusted accordingly to reach      the optimal value and the cost function is minimized. We use the new weights for prediction and to calculate the new cost. The calculation of the gradient and      the weight update till further adjustments to the weight do not reduce significantly. 

   We run by starting 1000 iterations with a learning rate of 0.05. We reach at an intercept value of(theta0) and a slope value of (theta1) with the final cost        function value. 
   A graph is plotted of the cost_history over iterations which shows the converged point.

4. Conjugate_Gradient_Method.py : Conjugate gradient method is used when the function to be minimized is smooth. Here, the function is optimized incrementally in an    iterative procedure. We start by specifying a function to be minimized (f1).For a smooth functioned to be minimized we have to take Jacobian vector of partial      derivates and for the hessian matrix which is implemented in df1 and df2 respectively. 

   A callback function is defined. This function will be called after each iteration. 
   An initial point to start is specified and conjugate gradient method from scipy.optimize library is imported.The iterations are plot using a plot.iteration          function.
   


    
    
    
 
    
