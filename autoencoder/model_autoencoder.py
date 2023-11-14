
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Convolutional layer 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling after Convolutional layer 1
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Convolutional layer 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling after Convolutional layer 2
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Convolutional layer 3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling after Convolutional layer 3            
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),  # Convolutional layer 4
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # Deconvolutional layer 4
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsampling
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Deconvolutional layer 3
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsampling
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Deconvolutional layer 2
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsampling
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Deconvolutional layer 1
            nn.Sigmoid()  # Sigmoid activation for output
        )
    def forward(self, x, mode):
        x2 = self.encoder(x)
        if mode == "autoencoder":
            x3 = self.decoder(x2)
            return x3
        else:
            return x2
