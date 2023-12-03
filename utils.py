import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset
import torch
from torchvision.io import read_image

# Function to plot loss
def plot_losses(train, test, path, name_plot, title, axis_x, axis_y, label_1, label_2):
    plt.figure(figsize=(10, 5))
    plt.plot(train[1:], label=label_1)
    plt.plot(test[1:], label=label_2)
    plt.xlabel(axis_x)
    plt.ylabel(axis_y)
    plt.legend()
    plt.title(title)
    plt.savefig(f"{path}{name_plot}.png")

def plot_confusion_matrix(target, predictions, path, name_plot):
    cm = confusion_matrix(target, predictions, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0,1])
    disp.plot()
    plt.savefig(f"{path}{name_plot}.png")


class ImagesDataset(Dataset):
    def __init__(self, data, targets, data_dir, transform=None):

        self.data_dir = data_dir
        self.target = targets
        self.data = data
        self.transforms = transform
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = read_image(self.data_dir+self.data[idx])[:-1,:,:]
        img = img.to(torch.float32)
        img = img/255
        target = self.target[idx]

        if self.transforms:
            img = self.transforms(img)

        if target == 0:
            target = [1,0]
        else:
            target = [0,1]
        return img, torch.Tensor(target)