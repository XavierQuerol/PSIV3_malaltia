import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),  # Convolutional layer 1
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Convolutional layer 2
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling after Convolutional layer 1
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # Convolutional layer 2
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),  # Convolutional layer 2
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling after Convolutional layer 1
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Convolutional layer 2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # Convolutional layer 2
            nn.Flatten(),

        )

        self.predict = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 16),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.predict(x)
        return x