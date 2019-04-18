import torch
import pandas as pd
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

id_to_classname = {574:'golf ball', 471:'cannon', 455:'bottlecap'}

class ClassificationDataset(Dataset):

    def __init__(self, images_root='day_10-example_data/images/', csv='day_10-example_data/labels.csv', transform=None):
        self.csv = pd.read_csv(csv)
        self.images_root=images_root
        self.fnames = self.csv['Filename'].tolist()
        self.labels = self.csv['Label'].tolist()
        self.transform = transform
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self,idx):
        filepath = self.images_root+self.fnames[idx]
        img = Image.open(filepath)
        label = self.labels[idx]
        if self.transform:
            img, label = self.transform((img, label))
        return img, label

class SquareResize():
    """Adjust aspect ratio of image to make it square"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)) # assert output_size is int or tuple
        self.output_size = output_size

    def __call__(self, sample):        
        image, label = sample
        h, w = image.size
        if h>w:
            new_w = self.output_size
            scale = new_w/w
            new_h = scale*h
        elif w>h:
            new_h = self.output_size
            scale = new_h/h
            new_w = scale*w
        else:
            new_h, new_w = self.output_size, self.output_size
        new_h, new_w = int(new_h), int(new_w) # account for non-integer computed dimensions (rounds to nearest int)
        image = image.resize((new_h, new_w))
        image = image.crop((0, 0, self.output_size, self.output_size))
        return image, label

class ToTensor():
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample
        image = np.array(image)/255
        image = image.transpose((2, 0, 1))
        return torch.Tensor(image), label

def test():
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    img_label_text = ax.text(0, -5, '', fontsize=15)
    print('Started evaluation...')
    mymodel.eval() #put model into evaluation mode
    
    #calculate the accuracy of our model over the whole test set in batches
    correct = 0
    for x, y in test_samples:
        h = mymodel.forward(x)
        pred = h.data.max(1)[1]
        correct += pred.eq(y).sum().item()

        y_ind=0
        im = np.array(x[y_ind])
        im = np.array(x[y_ind]).transpose(1, 2, 0)
        predicted_class = id_to_classname[h.max(1)[1][y_ind].item()]
        ax.imshow(im)
        img_label_text.set_text('Predicted class: '+ str(predicted_class))
        fig.canvas.draw()
        plt.pause(1)
                
    acc = round(correct/len(test_data), 4)
    print('Test accuracy', acc)
    return acc

mytransforms = []
mytransforms.append(SquareResize(224))
mytransforms.append(ToTensor())
mytransforms = transforms.Compose(mytransforms)

batch_size=1
test_data = ClassificationDataset(transform=mytransforms)
test_samples = DataLoader(test_data, batch_size=batch_size, shuffle=True)
mymodel = models.resnet18(pretrained=True)
test()

    
