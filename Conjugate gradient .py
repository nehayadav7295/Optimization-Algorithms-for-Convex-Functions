#!/usr/bin/env python
# coding: utf-8

# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, set_matplotlib_formats 
set_matplotlib_formats('pdf', 'svg') # toggle vector graphics f

f1 = lambda x : x**2

from scipy.optimize import minimize_scalar

res = minimize_scalar(f1, method='brent')
print('xmin: %.02f, fval: %.02f, iter: %d' % (res.x, res.fun, res.nit))


    


# In[20]:


x = np.linspace(res.x - 0.5, res.x + 0.5, 100)
y = [f1(val) for val in x]
plt.plot(x, y, color='blue', label='f1')

# plot optima
plt.scatter(res.x, res.fun, color='orange', marker='x', label='opt')

plt.grid()
plt.legend(loc=1)


# In[40]:


def plot_iterations(f, iterations, delta=0.1, box=1):
    xs = np.arange(-box, +box, delta)
    ys = np.arange(-box, +box, delta)
    X, Y = np.meshgrid(xs, ys)
    # create function mesh
    Z = np.array([[f(np.array([x, y])) for x in xs] for y in ys])
    # contour plot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    cd = ax.contour(X, Y, Z)
    #error plot
    ax.clabel(cd, inline=1, fontsize=10)
    ax.plot(iterations[:, 0], iterations[:, 1], color='red', marker='x', label='opt')
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(range(len(iterations)), [f(np.array(step)) for step in iterations], 'xb-')
    ax.grid('on')
    ax.set_xlabel('iteration')
    ax.set_xticks(range(len(iterations)))
    ax.set_ylabel('f(x)')


# In[41]:


def get_callback_function(data_list):
    def callback_function(x):
        data_list.append(x)
    return callback_function


# In[42]:


f2 = lambda x : x.T.dot(np.diag([2, 5])).dot(x)


# In[43]:


#partial derivative of function 2

def df2(x):
    #returns the vector of partial derivatives
    return (np.diag([2, 5]) + np.diag([2, 5]).T).dot(x)


# In[44]:


# second partial derivative
def ddf2(x):
    #returns the Hessian matrix of second partial derivatives
    return np.diag([4, 10])


# In[45]:


# specify initial point
x0 = [-1, 1]
# initialise callback data
cg_data = [np.array(x0)]
#imports the conjugate gradient method from scipy.optimize
from scipy.optimize import fmin_cg
#runs optimisation algorithm
res = fmin_cg(f2, x0, fprime=df2, callback=get_callback_function(cg_data))


# In[46]:


plot_iterations(f2, np.array(cg_data))


# In[ ]:




