import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(10)

#x is the input
#y is the label
#h is our prediction our model makes of the label
#m is the gradient of the straight line model
#b is the bias of the straigh line model with a bias
#cost a measure of how bad our model is, we are trying to minimize this

###grid search and random search over single parameter/straight line
#our model takes the form of a straight line passing through the origin
#h = mx
'''X = np.arange(0, 10, 0.5) #input features
Y = 3.2*X + 4.7 + np.random.randn(len(X)) #labels

n_search = 30 #hyperparameter #how many values should we search
search_min_val = -5 #hyperparameter
search_max_val = 5 #hyperparameter
#m_search = np.linspace(search_min_val, search_max_val, n_search) #grid search 
#m_search = np.random.rand(n_search)*(search_max_val-search_min_val) - (search_max_val-search_min_val)/2 #random search

plt.ion()
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_xlabel('m value')
ax1.set_ylabel('Cost')
ax1.set_xlim(-5, 5)
ax1.set_ylim(0, 3000)

ax2.set_title('Predictions vs labels')

costs = []
for m in m_search:
    h = m*X#hypthesis/prediction
    cost = np.mean((h-Y)**2)
    costs.append(cost)
    #print('Cost', cost)

    ax1.scatter([m], [cost], c='b')
    #ax1.scatter([min_cost], [min_m], c='r')

    ax2.scatter(X, Y)
    ax2.plot(X, h, 'g')

    fig.canvas.draw()
    plt.pause(0.5)

#min_ind = np.argmin(costs) #index of minimum value of y
#min_cost = costs[min_ind] #minimum value of y
#min_m = m_search[min_ind] #value of x for minimum value of y
#print('Lowest Cost of',min_cost, 'occurs at M=', min_m)'''

###gradient descent single parameter
#our model takes the form of a straight line passing through the origin
#h = mx
'''X = np.arange(0, 10, 0.5) #input features
Y = 0.8*X**2 + 3.2*X + 4.7 + np.random.randn(len(X)) #labels

#cost = (h-y)**2
#cost = (mx-y)**2
#cost gradient = 2*(mx-y) = 2x*(h-y)
def my_deriv(x, h, y):
    return 2*np.dot(x, (h-y))

#y = mx
learning_rate = 0.001 #hyperparameter #controls the size of the steps you take 
steps = 10 #hyperparameter #number of steps to take

#plotting stuff
plt.ion()
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_xlabel('m value')
ax1.set_ylabel('Cost')
ax1.set_xlim(-5, 5)
ax1.set_ylim(0, 3000)

ax2.set_title('Predictions vs labels')##

new_m = np.random.randn()
costs = []
for i in range(steps):
    current_m = new_m
    h = current_m*X#hypthesis/prediction
    cost = np.mean((h-Y)**2)
    costs.append(cost)
    deriv = my_deriv(X, h, Y)

    new_m = current_m - learning_rate*deriv
    #print('Cost', cost)

    print(current_m)
    print(cost)
    print()
    ax1.scatter([current_m], [cost], c='b')
    #ax1.scatter([min_cost], [min_m], c='r')

    ax2.scatter(X, Y)
    ax2.plot(X, h, 'g')

    fig.canvas.draw()
    plt.pause(0.5)

#min_ind = np.argmin(costs) #index of minimum value of y
#min_cost = costs[min_ind] #minimum value of y
#min_m = m_search[min_ind] #value of x for minimum value of y
#print('Lowest Cost of',min_cost, 'occurs at M=', min_m)'''
    

###gradient descent two variables variable.
#our model takes the form of a straight line model with a bias
#h = mx+b
'''X = np.arange(0, 10, 0.5) #input features
Y = 3.2*X + 4.7 + np.random.randn(len(X)) #labels

#cost = (h-y)**2
#cost = (mx+b-y)**2
#cost gradient m = 2x*(m+b-y) = 2x*(h-y)
#cost gradient b = 2(h-y)
def my_deriv(x, h, y):
    deriv_m = 2*np.dot(x, (h-y))/len(x)
    deriv_b = 2*np.sum(h-y)/len(x)
    return deriv_m, deriv_b

#y = mx + b
learning_rate = 0.001 #hyperparameter #controls the size of the steps you take 
steps = 100 #hyperparameter #number of steps to take

#plotting stuff
plt.ion()
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.set_xlabel('m value')
ax1.set_ylabel('Cost')
#ax1.set_xlim(-5, 5)
ax1.set_ylim(0, 3000)

ax2.set_xlabel('b value')
ax2.set_ylabel('Cost')
#ax2.set_xlim(-5, 5)
ax2.set_ylim(0, 3000)

ax3.set_title('Predictions vs labels')##

new_m = np.random.randn()
new_b = np.random.rand()*10-5
costs = []
for i in range(steps):
    current_m = new_m
    current_b = new_b
    h = current_m*X + current_b#hypthesis/prediction
    cost = np.mean((h-Y)**2)
    costs.append(cost)
    deriv_m, deriv_b = my_deriv(X, h, Y)

    new_m = current_m - learning_rate*deriv_m
    new_b = current_b - learning_rate*deriv_b
    #print('Cost', cost)

    print('Current M', current_m, 'Current B', current_b)
    print(cost)
    print()
    ax1.scatter([current_m], [cost], c='b')
    ax2.scatter([current_b], [cost], c='b')
    #ax1.scatter([min_cost], [min_m], c='r')

    ax3.scatter(X, Y)
    ax3.plot(X, h, 'g')

    fig.canvas.draw()
    plt.pause(0.5)

#min_ind = np.argmin(costs) #index of minimum value of y
#min_cost = costs[min_ind] #minimum value of y
#min_m = m_search[min_ind] #value of x for minimum value of y
#print('Lowest Cost of',min_cost, 'occurs at M=', min_m)
'''
