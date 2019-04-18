import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

train_data = datasets.MNIST('./mnist-data',
                           train=True,
                           transform=transforms.ToTensor(),
                           target_transform=None,
                           download=True) #60,000 train images

test_data = datasets.MNIST('./mnist-data',
                           train=False,
                           transform=transforms.ToTensor(),
                           target_transform=None,
                           download=True) #10,000 test images
#train_data, _ = torch.utils.data.random_split(train_data, [len(train_data)//24, 23*len(train_data)//24]) #we want to use a small subset of the actual data so we can finish epochs quickly to plot. in reality, you would would to train on all of the available data
train_size = int(0.8 * len(train_data)) #0.8*60000
val_size = len(train_data) - train_size ##0.2*60000
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

batch_size=32

train_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)           # dataloaders batch and shuffle the dataset for us
val_samples = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)

class ResBlock(torch.nn.Module):
    def __init__(self, n_layers=128):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(n_layers, n_layers, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_layers, n_layers, kernel_size=3, stride=1, padding=1),
            )
    def forward(self, x):
        return x+self.block(x)


class MNIST_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()                               # define the actuvation function you want to use here (dont create it by calling it)
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), #32x14x14
            torch.nn.ReLU(),                                           # create the activation function at this layer
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), #64x7x7
            torch.nn.ReLU(),                                       # create the activation function at this layer
            ResBlock(64),
            torch.nn.ReLU(),
            ResBlock(64),
            torch.nn.ReLU(),
        )
        self.dense_layers = torch.nn.Sequential(
            torch.nn.Linear(64*7*7, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 10),
            torch.nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_layers(x).view(-1, 64*7*7)
        return self.dense_layers(x)

classnames = map(str, list(range(10)))

lr = 3e-4

mymodel = MNIST_CNN()
optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr)

def train(epochs, pause_len=0.5):
    plt.close()
    mymodel.train()
    
    bcosts = []
    ecosts = []
    valcosts = []
    plt.ion()
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    
    plt.show()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')

    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Cost')

    ax2.axis('off')
    img_label_text = ax2.text(0, -5, '', fontsize=15)
    
    for e in range(epochs):
        ecost=0
        valcost=0
        for i, (x, y) in enumerate(train_samples):

            h = mymodel.forward(x) #calculate hypothesis
            cost = F.cross_entropy(h, y, reduction='sum') #calculate cost
            
            optimizer.zero_grad() #zero gradients
            cost.backward() # calculate derivatives of values of filters
            optimizer.step() #update parameters

            bcosts.append(cost.item()/batch_size)
            ax1.plot(bcosts, 'b', label='Train cost')
            if e==0 and i==0: ax1.legend()
            
            y_ind=0
            im = np.array(x[y_ind][0])
            predicted_class = h.max(1)[1][y_ind].item()
            ax2.imshow(im)
            img_label_text.set_text('Predicted class: '+ str(predicted_class))
            
            fig.canvas.draw()
            ecost+=cost.item()
        for i, (x, y) in enumerate(val_samples):

            h = mymodel.forward(x) #calculate hypothesis
            cost = F.cross_entropy(h, y, reduction='sum') #calculate cost

            '''for y_ind, yval in enumerate(y):
                if yval.item() not in classes_shown:
                    classes_shown.add(yval.item())
                    break'''
            y_ind=0
            im = np.array(x[y_ind][0])
            predicted_class = h.max(1)[1][y_ind].item()
            ax2.imshow(im)
            img_label_text.set_text('Predicted class: '+ str(predicted_class))
            fig.canvas.draw()
            if pause_len!=0:
                plt.pause(pause_len)
            
            valcost+=cost.item()
        ecost/=train_size
        valcost/=val_size
        ecosts.append(ecost)
        valcosts.append(valcost)
        ax.plot(ecosts, 'b', label='Train cost')
        ax.plot(valcosts, 'r', label='Validation cost')
        if e==0: ax.legend()
        fig.canvas.draw()

        print('Epoch', e, '\tCost', ecost)

def test():
    print('Started evaluation...')
    mymodel.eval() #put model into evaluation mode
    
    #calculate the accuracy of our model over the whole test set in batches
    correct = 0
    for x, y in test_samples:
        h = mymodel.forward(x)
        pred = h.data.max(1)[1]
        correct += pred.eq(y).sum().item()
    acc = round(correct/len(test_data), 4)
    print('Test accuracy', acc)
    return acc

train(epochs=1, pause_len=0)
test()
