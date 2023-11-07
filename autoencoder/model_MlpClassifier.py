import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.Linear(128, 128),
            nn.Linear(128, 16),
            nn.Linear(16, output_dim)
        )
            
    def forward(self, x):
        x = self.encoder(x)
        return x