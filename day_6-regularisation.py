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
train_data, _ = torch.utils.data.random_split(train_data, [len(train_data)//24, 23*len(train_data)//24])
train_size = int(0.8 * len(train_data)) #0.8*60000
val_size = len(train_data) - train_size ##0.2*60000
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

batch_size=32

train_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)           # dataloaders batch and shuffle the dataset for us
val_samples = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)

im, label = train_data[0] #images are 28x28, label is digit in image

#print(im) 
#print(label)
#plt.imshow(im.numpy()[0])
#plt.show()

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()                                  # initialise the parent class
        ac = torch.nn.Sigmoid                               # define the actuvation function you want to use here (dont create it by calling it)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(28*28, 200),                         # linear layer from 1 unit to 50 (make 50 linear combinations of this input)
            torch.nn.ReLU(),                                           # create the activation function at this layer
            torch.nn.Linear(200, 100),                        # hidden layer
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 10),                         # output layer (50 units -> single scalar output prediction)
            torch.nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)                               # forward pass

nn = NN()

optimiser = torch.optim.Adam(nn.parameters(), lr=0.001, weight_decay=1e-4)           # create stochastic gradient descent optimiser
criterion = torch.nn.MSELoss()                                  # create function to

epochs = 100

fig = plt.figure(figsize=(10, 5))
loss_ax = fig.add_subplot(121)
loss_ax.set_ylabel('Loss')
loss_ax.set_xlabel('Batch')
epoch_loss_ax = fig.add_subplot(122)
epoch_loss_ax.set_ylabel('Loss')
epoch_loss_ax.set_xlabel('Epoch')
epoch_losses = []
epoch_val_losses = []
for epoch in range(epochs):                                                         # how many times to go through whole dataset?
    batch_losses = []
    loss_ax.clear()
    for batch_idx, (x, y) in enumerate(train_samples):                                  # for each minibatch from dataloader
        x = x.view(-1, 28*28)                                                           # unpack the minibatch
        h = nn(x)                                                                  # make predictions for this minibatch
        loss = F.cross_entropy(h, y)                                                     # evaluate loss for this batch
        loss.backward()                   # differentiate loss with respect to parameters that the optimiser is tracking
        optimiser.step()                                                            # take optimisation step
        optimiser.zero_grad()                                                       # set parameter gradients = 0 (otherwise they accumulate)

        # PURELY FOR PLOTTING THE HYPOTHESIS OVER THE WHOLE INPUT DOMAIN
        #all_preds = nn(X).detach().numpy()                  # make predictions for all inputs (not just minibatch)
        #h_plot = h_ax.plot(X.numpy(), all_preds, c='g')    # plot predictions for all inputs

        print(f'Epoch: {epoch} \t\t Batch: {batch_idx} \t\tLoss: {loss.item()}')
        
        batch_losses.append(loss.item())
        loss_ax.plot(batch_losses, 'b')
        fig.canvas.draw()

        plt.pause(0.1)                                      # pause to give us time to view the hypothesis plot
    val_losses = []
    for batch_idx, (x, y) in enumerate(train_samples):                                  # for each minibatch from dataloader
        x = x.view(-1, 28*28)                                                           # unpack the minibatch
        h = nn(x)                                                                  # make predictions for this minibatch
        loss = F.cross_entropy(h, y)
        val_losses.append(loss.item())
        
    epoch_val_losses.append(np.mean(val_losses))
    epoch_losses.append(np.mean(batch_losses))
    epoch_loss_ax.plot(epoch_losses, 'b', label='Train')
    epoch_loss_ax.plot(epoch_val_losses, 'r', label='Val')
    if epoch==0: epoch_loss_ax.legend()
    fig.canvas.draw()
