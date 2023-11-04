
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),  # Convolutional layer 1
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Convolutional layer 2
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Convolutional layer 3
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Deconvolutional layer 1
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Deconvolutional layer 2
            nn.ReLU(),
            nn.ConvTranspose2d(64, 4, kernel_size=4, stride=2, padding=1),  # Deconvolutional layer 3
            nn.Sigmoid()  # Sigmoid activation for output
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
