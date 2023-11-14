import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.Linear(256, 128),
            nn.Linear(128, 16),
            nn.Linear(16, 1)
        )
            
    def forward(self, x):
        x = self.encoder(x)
        x = torch.sigmoid(x)
        return x