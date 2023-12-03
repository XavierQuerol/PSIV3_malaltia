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
    transforms.Resize((64, 64), antialias=True),  # Adjust mean and std as needed
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64), antialias=True),  # Adjust mean and std as needed
    transforms.Normalize(mean=[0.5], std=[0.5])
])

metadata = pd.read_csv(METADATA_FILE)
directories = [dir.path for dir in os.scandir(CROPPED_PATCHES_DIR) if dir.is_dir() and
               metadata.loc[metadata["CODI"] == dir.path.split("/")[-1].split("_")[0], "DENSITAT"].values[0] == "NEGATIVA"]

files = [os.path.join(directory.split("/")[-1], file) for directory in directories for file in os.listdir(directory)]
random.shuffle(files)
files = files[:15000]
data = files[:int(len(files)*0.8)]
targets = [0 for d in range(len(data))]
train_dataset = ImagesDataset(data=data, targets=targets, data_dir=CROPPED_PATCHES_DIR, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

data = files[int(len(files)*0.8):]
targets = [0 for d in range(len(data))]
test_dataset = ImagesDataset(data=data, targets=targets, data_dir=CROPPED_PATCHES_DIR, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No need to shuffle for testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
    
# Applying it to our net
model.apply(initialize_weights)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=1/3)

train_losses = []
test_losses = []

plot_interval = 10  # Define the interval for plotting losses
num_epochs = 15

best_test_loss  = 999

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
    plot_losses(train=train_losses, test=test_losses, path=PLOT_LOSS_DIR, name_plot=f"main_autoencoder_losses",
                    title="Loss over epoch", axis_x="Epoch", axis_y="Loss", label_1="Train losses", label_2="Test losses")

    scheduler.step(average_loss_test)
    # Save the trained model if needed
    if average_loss_test < best_test_loss:
        best_test_loss = average_loss_test
        torch.save(model.state_dict(), f'{SAVE_MODEL_DIR}model5_AUTOENCODER.pth')