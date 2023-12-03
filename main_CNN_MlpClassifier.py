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
import random

from ..baseline import model_autoencoder
from model_CNN_MlpClassifier import Classifier
from torchvision import models
from utils import ImagesDataset
from utils import plot_losses, plot_confusion_matrix

from config import *

model_to_train = "resnet"

metadata = pd.read_csv(METADATA_FILE)
window_metadata = pd.read_csv(WINDOW_METADATA_FILE)

directories = [dir.path for dir in os.scandir(ANNOTATED_PATCHES_DIR) if dir.is_dir()]

# Define transformations and data loaders for training and testing
train_transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomVerticalFlip(p=0.5),  
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # Adjust distortion_scale as needed
    transforms.RandomRotation(degrees=(0, 8)),  # Rotate in all angles
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust mean and std as needed
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust mean and std as needed
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
                    files.append(d+'/'+p)
            except:
                continue

combined = list(zip(files, targets))

# Shuffle the combined list
random.shuffle(combined)

# Extract the shuffled lists
files, targets = zip(*combined)
data = files[:int(len(files)*0.8)]
targets_train = targets[:int(len(files)*0.8)]
train_dataset = ImagesDataset(data=data, targets=targets_train, data_dir=ANNOTATED_PATCHES_DIR, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

data = files[int(len(files)*0.8):]
targets_test = targets[int(len(files)*0.8):]
test_dataset = ImagesDataset(data=data, targets=targets_test, data_dir=ANNOTATED_PATCHES_DIR, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # No need to shuffle for testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)

if model_to_train == "our_classifier":
    model = Classifier()
    model.apply(initialize_weights)

elif model_to_train == "resnet":
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),  # Replace the last fully connected layer with your custom layers
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 2)  # Assuming your final output has 2 classes, modify this as needed
    )
    model.fc.apply(initialize_weights)

    for param in model.layer4.parameters():
        param.requires_grad = True

model.to(device)

ones = sum(targets[:int(len(files)*0.8)])/len(targets[:int(len(files)*0.8)])
zeros = 1 - ones

criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, zeros/ones])).to(device)
#criterion = nn.BCELoss(weight=torch.Tensor(1000))
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=1/3)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

best_test_loss = 999

num_epochs = 200

for epoch in range(num_epochs):
    # Training loop
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    predicted_train = []
    labels_train = []
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).reshape([len(images), -1])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predicted = torch.argmax(torch.sigmoid(outputs), dim=1)
        labels = torch.argmax(labels, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_train += predicted.tolist()
        labels_train += labels.tolist()

        
    # Calculate average loss over the dataset
    average_loss = total_loss / len(train_dataloader)
    accuracy = 100 * correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Testing loop (outside the training loop)
    model.eval()
    total_loss_test = 0
    correct = 0
    total = 0
    predicted_test = []
    labels_test = []
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            test_outputs = model(images).reshape([len(images), -1])
            test_loss = criterion(test_outputs, labels)
            total_loss_test += test_loss.item()

            # Calculate accuracy

            predicted = torch.argmax(torch.sigmoid(test_outputs), dim=1)
            labels = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_test += predicted.tolist()
            labels_test += labels.tolist()

    # Calculate average test loss and accuracy
    average_loss_test = total_loss_test / len(test_dataloader)
    accuracy_test = 100 * correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {average_loss_test:.4f}, Accuracy: {accuracy_test:.2f}%')

    # Append losses for plotting
    train_losses.append(average_loss)
    test_losses.append(average_loss_test)
    train_accuracies.append(accuracy)
    test_accuracies.append(accuracy_test)

    # Plot final loss
    if epoch != 0:
        plot_losses(train=train_losses, test=test_losses, path=PLOT_LOSS_DIR, name_plot=f"main_{model_to_train}_losses",
                    title="Loss over epoch", axis_x="Epoch", axis_y="Loss", label_1="Train losses", label_2="Test losses")

        plot_losses(train=train_accuracies, test=test_accuracies, path=PLOT_LOSS_DIR, name_plot=f"main_{model_to_train}_accuracy",
                    title="Accuracy over epoch", axis_x="Epoch", axis_y="Accuracy", label_1="Train accuracy", label_2="Test accuracy")
        
        plot_confusion_matrix(target=labels_train, predictions=predicted_train, path=PLOT_LOSS_DIR, name_plot=f"main_{model_to_train}_MC_train")
        plot_confusion_matrix(target=labels_test, predictions=predicted_test, path=PLOT_LOSS_DIR, name_plot=f"main_{model_to_train}_MC_test")
                    

    scheduler.step(average_loss_test)
    print(optimizer.param_groups[0]['lr'])

    if average_loss_test < best_test_loss:
        best_test_loss = average_loss_test
        torch.save(model.state_dict(), f'{SAVE_MODEL_DIR}model_MLP{model_to_train}.pth')
# Save the trained model if needed
