import matplotlib.pyplot as plt

# Function to plot loss
def plot_losses(train_losses, test_losses, path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Autoencoder Training and Testing Loss')
    plt.savefig(f"{path}loss.png")