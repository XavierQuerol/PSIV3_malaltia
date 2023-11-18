import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Convolutional layer 1
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling after Convolutional layer 1
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Convolutional layer 2
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling after Convolutional layer 2
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # Convolutional layer 3
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling after Convolutional layer 3            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Convolutional layer 4
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # Deconvolutional layer 4
            nn.ReLU(),
            #nn.MaxUnpool2d(kernel_size=2, stride=2),  # Max unpooling
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),  # Deconvolutional layer 3
            nn.ReLU(),
            #nn.MaxUnpool2d(kernel_size=2, stride=2),  # Max unpooling
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),  # Deconvolutional layer 2
            nn.ReLU(),
            #nn.MaxUnpool2d(kernel_size=2, stride=2),  # Max unpooling
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),  # Deconvolutional layer 1
            nn.Sigmoid()  # Sigmoid activation for output
        )


    """def forward(self, x, mode):
        x2 = self.encoder(x)
        if mode == "autoencoder":
            x3 = self.decoder(x2)
            return x3
        else:
            return x2"""

    def forward(self, x, mode):
            # Encoder
            x, indices1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)(self.encoder[0](x))
            x = self.encoder[1](x)
            x, indices2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)(self.encoder[2](x))
            x = self.encoder[3](x)
            x, indices3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)(self.encoder[4](x))
            x = self.encoder[5](x)
            x, indices4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)(self.encoder[6](x))
            x = self.encoder[7](x)

            if mode == "autoencoder":
                x = nn.MaxUnpool2d(kernel_size=2, stride=2)(x, indices4)
                x = self.decoder[1](self.decoder[0](x))
                x = nn.MaxUnpool2d(kernel_size=2, stride=2)(x, indices3)
                x = self.decoder[3](self.decoder[2](x))
                x = nn.MaxUnpool2d(kernel_size=2, stride=2)(x, indices2)
                x = self.decoder[5](self.decoder[4](x))
                x = nn.MaxUnpool2d(kernel_size=2, stride=2)(x, indices1)
                x = self.decoder[7](self.decoder[6](x))
                
            return x
