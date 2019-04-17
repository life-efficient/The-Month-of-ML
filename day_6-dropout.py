import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

unloader = transforms.ToPILImage()

train_data = datasets.MNIST(root='.', train=True, transform=transforms.ToTensor(), download=True)
val_data = datasets.MNIST(root='.', train=False, transform=transforms.ToTensor())

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=4)

class NN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(784, 30),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(30, 30),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(30, 10),
            torch.nn.Softmax()
        )

    def forward(self, x):
        x = x.view(-1, 1*28*28)
        h = self.layers(x)
        return h

nn = NN()

criterion = torch.nn.NLLLoss()
optimiser = torch.optim.Adam(nn.parameters(), lr=0.0001)

def train_epoch(epoch):
    for batch_idx, batch in enumerate(train_dataloader):
        features, labels = batch
        pred = nn(features)

        loss = criterion(pred, labels)
        loss.backward()

        optimiser.step()
        optimiser.zero_grad()

        print('Epoch:', epoch, '\tBatch:', batch_idx, '\tLoss:', loss.item())

        if batch_idx == 200:
            #break
            pass

def evaluate():
    correct = 0
    for batch in val_dataloader:
        features, labels = batch
        pred = nn(features)
        pred = pred.detach().numpy()
        pred = np.argmax(pred, axis=1)
        eq = pred == labels.detach().numpy()
        correct += eq.sum()

    print('Accuracy:', correct / len(val_data))

epochs = 1
for epoch in range(epochs):
    train_epoch(epoch)
    evaluate()






