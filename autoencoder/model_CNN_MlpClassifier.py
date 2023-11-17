import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=1),  # Convolutional layer 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4),  # Max pooling after Convolutional layer 1
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Convolutional layer 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4),  # Max pooling after Convolutional layer 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Convolutional layer 3
            nn.ReLU(),
        )

        self.predict = nn.Sequential(
            nn.Linear(256, 16),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.predict(x)
        x = torch.sigmoid(x)
        return x