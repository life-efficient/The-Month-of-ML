import numpy as np
import matplotlib.pyplot as plt

### Uncomment each section and play with the variables marked as hyper-parameters
### 

###grid search and random search
'''
n_search = 10 #hyperparameter #how many values should we search

x_plot = np.linspace(0.001, 2.5, 400)
y_plot = (np.sin(10*np.pi*x_plot)/(2*x_plot)) + (x_plot-1)**4 #function we are optimizing

x = np.arange(0.001, 2.5, 2.5/n_search) #grid search 
#x = np.random.rand(10)*2.5 #random search

y = (np.sin(10*np.pi*x)/(2*x)) + (x-1)**4
min_ind = np.argmin(y) #index of minimum value of y
min_y = y[min_ind] #minimum value of y
min_x = x[min_ind] #value of x for minimum value of y

print('Lowest Y of',min_y, 'occurs at X', min_x)
plt.figure()
plt.plot(x_plot, y_plot)
plt.scatter(x, y, c='b')
plt.scatter([min_x], [min_y], c='r')
plt.show()
'''


###gradient descent with known equation
'''
def my_function(x, deriv=False):
    if deriv:
        return (-np.sin(10*np.pi*x)/(2*x**2)) + 4*(x-1)**3 + (5*np.pi*np.cos(10*np.pi*x))/x
    else:
        return (np.sin(10*np.pi*x)/(2*x)) + (x-1)**4

learning_rate = 0.001 #hyperparameter #controls the size of the steps you take 
steps = 10 #hyperparameter#number of steps to take

x_plot = np.linspace(0.001, 2.5, 400)
y_plot = my_function(x_plot)

plt.ion()
plt.figure()
plt.plot(x_plot, y_plot)
new_x = (np.random.rand()*2)+0.001
for i in range(steps):
    current_x = new_x
    current_y = my_function(current_x)
    current_deriv = my_function(current_x, deriv=True)
    new_x = current_x - learning_rate*current_deriv
    #final position of search is plotted in red
    plt.scatter([current_x], [current_y], c='r' if i==steps-1 else 'b')
'''

###gradient descent without known equation
###we can sample from the function but we cant model it mathematically
'''def my_function(x):
    #pretend we dont know this equation
    return (np.sin(10*np.pi*x)/(2*x)) + (x-1)**4

learning_rate = 0.001 #hyperparameter #controls the size of the steps you take 
steps = 10 #hyperparameter #number of steps to take
epsilon = 0.001 #hyperparameter #step size for calculating gradient, lower is better

x_plot = np.linspace(0.001, 2.5, 400)
y_plot = my_function(x_plot)

plt.ion()
plt.figure()
plt.plot(x_plot, y_plot)
plt.show()
new_x = (np.random.rand()*2)+0.001
for i in range(steps):
    current_x = new_x
    current_y = my_function(current_x)
    #current_deriv = my_function(current_x, deriv=True)
    test_x = current_x + epsilon
    test_y = my_function(test_x)
    current_deriv = (test_y-current_y) / epsilon
    new_x = current_x - learning_rate*current_deriv

    #final position of search is plotted in red
    plt.scatter([current_x], [current_y], c='r' if i==steps-1 else 'b')
'''

###gradient descent with momentum
'''
def my_function(x, deriv=False):
    if deriv:
        return (-np.sin(10*np.pi*x)/(2*x**2)) + 4*(x-1)**3 + (5*np.pi*np.cos(10*np.pi*x))/x
    else:
        return (np.sin(10*np.pi*x)/(2*x)) + (x-1)**4

learning_rate = 0.001 #hyperparameter #controls the size of the steps you take 
steps = 10 #hyperparameter #number of steps to take
momentum_decay = 0.8 #hyperparameter #controls how much previous gradients contribute to the current step

x_plot = np.linspace(0.001, 2.5, 400)
y_plot = my_function(x_plot)

plt.ion()
plt.figure()
plt.plot(x_plot, y_plot)
plt.show()
new_x = (np.random.rand()*2)+0.001
current_momentum = 0
for i in range(steps):
    current_x = new_x
    current_y = my_function(current_x)
    current_deriv = my_function(current_x, deriv=True)
    current_momentum = (momentum_decay*current_momentum) + learning_rate*current_deriv
    new_x = current_x - current_momentum

    #final position of search is plotted in red
    plt.scatter([current_x], [current_y], c='r' if i==steps-1 else 'b')
'''



















