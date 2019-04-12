import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 10)
print(x)
y = 0.1*x**3 - x**2 + 25 + np.random.rand(10)


class Model():

    def __init__(self, lr=0.000001):
        self.lr = lr
        self.x_3 = np.random.rand()
        self.x_2 = np.random.rand()
        self.b = np.random.rand()
        print('x**3:', self.x_3, 'x**2:', self.x_2, 'bias:', self.b)

    def forward(self, x):
        h = self.x_3 * x**3 + self.x_2 * x**2 + self.b
        return h

    def update_weights(self, grad):
        self.x_3 -= self.lr * grad[0]
        self.x_2 -= self.lr * grad[1]
        self.b -= self.lr * grad[2]

mymodel = Model()
h = mymodel.forward(x)

def MSE_loss(pred, labels, grad=False, input=None):

    if grad:
        g_x3 = 2 * np.sum( (pred - labels) * input**3 )
        g_x2 = 2 * np.sum( (pred - labels) * input**2 )
        g_b = 2 * np.sum( (pred - labels) )
        return [g_x3, g_x2, g_b]

    loss = np.sum((pred - labels)**2) / len(pred)
    return loss

fig = plt.figure()
h_ax = fig.add_subplot(111)
h_ax.scatter(x, y)
plt.ion()
plt.show()

epochs = 100
for epoch in range(epochs):
    h = mymodel.forward(x)

    loss = MSE_loss(h, y)
    grads = MSE_loss(h, y, grad=True, input=x)
    print(grads)
    mymodel.update_weights(grads)

    lines = h_ax.plot(x, h)
    fig.canvas.draw()
    plt.pause(0.1)
    lines.pop(0).remove()







