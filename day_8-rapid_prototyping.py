import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from getDataLoaders import getDataLoaders
from training import train, evaluate
from profiler import Profiler

p = Profiler()

p.start('making fake data')
X = np.linspace(-10, 10)
Y = X**2 + 10*np.sin(X)
p.stop('making fake data')
p.show()

X = np.linspace(-10, 10)
Y = X**2 + 10*np.sin(X)

fig = plt.figure(figsize=(10, 5))
h_ax = fig.add_subplot(121)
loss_ax = fig.add_subplot(122)
loss_ax.set_ylim(0, 2500)

print(X, Y)
h_ax.plot(X, Y)
h_ax.set_ylim(-10, 50)

X = torch.tensor(X).float().unsqueeze(1)
Y = torch.tensor(Y).float().unsqueeze(1)
print(X)

plt.ion()
plt.show()

epochs = 10
batch_size = 24

class MyDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):                           # what to do when the instance is indexed with square brackets
        x = self.features[idx]                            # get features at that index
        x = torch.tensor(x)                               # convert numpy array to torch tensor so gradients are
        x = x.unsqueeze(0)
        y = self.labels[idx]
        y = torch.tensor(y)
        y = y.unsqueeze(0)
        return x, y

    def __len__(self):
        return len(self.features)                             # length of dataset is length of features

dataset = MyDataset(X, Y)                                                       # create an instance of the dataset

''' -----OLD-----
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)           # dataloaders batch and shuffle the dataset for us
val_loader = DataLoader(dataset, batch_size=batch_size)           # dataloaders batch and shuffle the dataset for us
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)           # dataloaders batch and shuffle the dataset for us
'''

# ----- NEW -----
train_loader, val_loader, test_loader = getDataLoaders(dataset, batch_size=4, splits=[0.75, 0.15, 0.1])
# ---------------

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()                                  # initialise the parent class
        ac = torch.nn.Sigmoid                               # define the actuvation function you want to use here (dont create it by calling it)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 30),                         # linear layer from 1 unit to 50 (make 50 linear combinations of this input)
            ac(),                                           # create the activation function at this layer
            torch.nn.Linear(30, 50),                        # hidden layer
            ac(),
            torch.nn.Linear(50, 50),                        # hidden layer
            ac(),
            torch.nn.Linear(50, 1)                          # output layer (50 units -> single scalar output prediction)
        )

    def forward(self, x):
        return self.layers(x)                               # forward pass

nn = NN()

optimiser = torch.optim.SGD(nn.parameters(), lr=0.01)           # create stochastic gradient descent optimiser
criterion = torch.nn.MSELoss()                                  # create function to

epochs = 100

train_losses = []
val_losses = []

for epoch in range(epochs):    # how many times to go through whole dataset?
    # ----- PURELY FOR PLOTTING THE HYPOTHESIS OVER THE WHOLE INPUT DOMAIN --------
    all_preds = nn(X).detach().numpy()                  # make predictions for all inputs (not just minibatch)
    h_plot = h_ax.plot(X.numpy(), all_preds, c='g')    # plot predictions for all inputs
    fig.canvas.draw()
    # -----------------------------------------------------------------------------

    train(nn, train_loader, criterion, optimiser, epoch, fig, loss_ax, train_losses, p)
    idx = len(train_loader) * epoch                     # index of current batch
    evaluate(nn, val_loader, criterion, epoch, fig, loss_ax, val_losses, idx)

    h_plot.pop(0).remove()                              # remove the previous plot


    '''
    #   --------- OLD ----------
    for batch_idx, batch in enumerate(train_loader):                                  # for each minibatch from dataloader
        print(batch)

        x, y = batch                                                               # unpack the minibatch
        h = nn(x)                                                                  # make predictions for this minibatch
        loss = criterion(h, y)                                                     # evaluate loss for this batch
        loss.backward()                   # differentiate loss with respect to parameters that the optimiser is tracking

        optimiser.step()                                                            # take optimisation step
        optimiser.zero_grad()                                                       # set parameter gradients = 0 (otherwise they accumulate)

        # PURELY FOR PLOTTING THE HYPOTHESIS OVER THE WHOLE INPUT DOMAIN
        all_preds = nn(X).detach().numpy()                  # make predictions for all inputs (not just minibatch)
        h_plot = h_ax.plot(X.numpy(), all_preds, c='g')    # plot predictions for all inputs

        print(f'Epoch: {epoch} \t\t Batch: {batch_idx} \t\tLoss: {loss.item()}')

        losses.append(loss.item())
        loss_ax.plot(losses, 'g')


        plt.pause(0.1)                                      # pause to give us time to view the hypothesis plot
        h_plot.pop(0).remove()                              # remove the previous plot
    '''