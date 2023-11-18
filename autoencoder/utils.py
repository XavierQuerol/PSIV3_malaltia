import matplotlib.pyplot as plt

# Function to plot loss
def plot_losses(train_losses, test_losses, path, name, name1, name2):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses[1:], label=name1)
    plt.plot(test_losses[1:], label=name2)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Autoencoder Training and Testing Loss')
    plt.savefig(f"{path}{name}.png")