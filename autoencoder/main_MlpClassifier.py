import torch
import os
import cv2
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image

from model_autoencoder import Autoencoder
from model_MlpClassifier import Classifier
from dataset import ImagesDataset
from utils import plot_losses

from config import *

model = Autoencoder()
model.load_state_dict(torch.load(f'{SAVE_MODEL_DIR}model_AUTOENCODER.pth'))

metadata = pd.read_csv(METADATA_FILE)
window_metadata = pd.read_csv(WINDOW_METADATA_FILE)

directories = [dir.path for dir in os.scandir(ANNOTATED_PATCHES_DIR) if dir.is_dir()]

# Define transformations and data loaders for training and testing
train_transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std as needed
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std as needed
])

metadata = pd.read_csv(WINDOW_METADATA_FILE)
targets = []
files = []
metadata[['IDPacient','IDWindow']] = metadata['ID'].str.split('.', n=1, expand=True)
for d in os.listdir(ANNOTATED_PATCHES_DIR):
    if d in list(metadata['IDPacient']):
        for p in os.listdir(ANNOTATED_PATCHES_DIR+'/'+d):
            try:
                id = d+'.'+p[:-4]
                target = int(metadata[metadata['ID'] == id]['Presence'])
                if target == -1:
                    targets.append(0)
                    files.append(d+'/'+p)
                elif target == 1:
                    targets.append(1)
                    files.append(+d+'/'+p)
            except:
                continue


data = files[:int(len(files)*0.8)]
targets_train = targets[:int(len(files)*0.8)]
train_dataset = ImagesDataset(data=data, targets=targets_train, data_dir=ANNOTATED_PATCHES_DIR, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

data = files[int(len(files)*0.8):]
targets_test = targets[int(len(files)*0.8):]
test_dataset = ImagesDataset(data=data, targets=targets_test, data_dir=ANNOTATED_PATCHES_DIR, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No need to shuffle for testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

input_dim = 256

model2 = Classifier(input_dim).to(device)

ones = sum(targets)/len(targets)
zeros = 1 - ones
pos_weight = zeros / (1 - zeros)

criterion = nn.BCELoss(torch.Tensor([pos_weight]))
optimizer = optim.Adam(model2.parameters(), lr=0.001)

train_losses = []
test_losses = []

plot_interval = 10  # Define the interval for plotting losses
num_epochs = 5

for epoch in range(num_epochs):
    # Training loop
    model.eval()
    model2.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            features = model(images, "encoder").reshape([-1, input_dim])
        outputs = model2(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Calculate average loss over the dataset
    average_loss = total_loss / len(train_dataloader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_loss:.4f}')

    # Testing loop (outside the training loop)
    model2.eval()
    total_loss_test = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            test_outputs = model(images, "encoder").reshape([-1, input_dim])
            test_outputs = model2(test_outputs)
            test_loss = criterion(test_outputs, labels)
            total_loss_test += test_loss.item()

            # Calculate accuracy
            predicted = torch.where(test_outputs > 0.5, torch.tensor(1), torch.tensor(0))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average test loss and accuracy
    average_loss_test = total_loss_test / len(test_dataloader)
    accuracy = 100 * correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {average_loss_test:.4f}, Accuracy: {accuracy:.2f}%')

    # Append losses for plotting
    train_losses.append(average_loss)
    test_losses.append(average_loss_test)

    # Plot final loss
    if (epoch + 1) % plot_interval == 0:
        plot_losses(train_losses, test_losses, PLOT_LOSS_DIR)

# Save the trained model if needed
torch.save(model.state_dict(), f'{SAVE_MODEL_DIR}model_MLP.pth')