import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from dataset import ImagesDataset
from model_autoencoder import Autoencoder
from config import METADATA_FILE, CROPPED_PATCHES_DIR, SAVE_MODEL_DIR, PLOT_LOSS_DIR
import os
import random
import pandas as pd
from utils import plot_losses


# Define transformations and data loaders for training and testing
train_transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std as needed
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std as needed
])

metadata = pd.read_csv(METADATA_FILE)
directories = [dir.path for dir in os.scandir(CROPPED_PATCHES_DIR) if dir.is_dir() and
               metadata.loc[metadata["CODI"] == dir.path.split("/")[-1].split("_")[0], "DENSITAT"].values[0] == "NEGATIVA"]

files = [os.path.join(directory.split("/")[-1], file) for directory in directories for file in os.listdir(directory)]
random.shuffle(files)
files = files[:200]
data = files[:int(len(files)*0.8)]
targets = [0 for d in range(len(data))]
train_dataset = ImagesDataset(data=data, targets=targets, data_dir=CROPPED_PATCHES_DIR, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

data = files[int(len(files)*0.8):]
targets = [0 for d in range(len(data))]
test_dataset = ImagesDataset(data=data, targets=targets, data_dir=CROPPED_PATCHES_DIR, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No need to shuffle for testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
test_losses = []

plot_interval = 10  # Define the interval for plotting losses
num_epochs = 5

for epoch in range(num_epochs):
    # Training loop
    model.train()
    total_loss = 0
    for batch_idx, (images,_) in enumerate(train_dataloader):  # No labels are needed
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images, "autoencoder")
        loss = criterion(outputs, images)  # Compare the output to the input
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Calculate average loss over the dataset
    average_loss = total_loss / len(train_dataloader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_loss:.4f}')

    # Testing loop (outside the training loop)
    model.eval()
    total_loss_test = 0
    with torch.no_grad():
        for images,_ in test_dataloader:  # No labels are needed
            images = images.to(device)
            test_outputs = model(images, "autoencoder")
            test_loss = criterion(test_outputs, images)
            total_loss_test += test_loss.item()

    # Calculate average test loss
    average_loss_test = total_loss_test / len(test_dataloader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {average_loss_test:.4f}')

    # Append losses for plotting
    train_losses.append(average_loss)
    test_losses.append(average_loss_test)

    # Plot final loss
    plot_losses(train_losses, test_losses, PLOT_LOSS_DIR)


# Save the trained model if needed
torch.save(model.state_dict(), f'{SAVE_MODEL_DIR}model_AUTOENCODER.pth')