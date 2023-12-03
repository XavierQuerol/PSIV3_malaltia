import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to plot loss
def plot_losses(train, test, path, name_plot, title, axis_x, axis_y, label_1, label_2):
    plt.figure(figsize=(10, 5))
    plt.plot(train[1:], label=label_1)
    plt.plot(test[1:], label=label_2)
    plt.xlabel(axis_x)
    plt.ylabel(axis_y)
    plt.legend()
    plt.title(title)
    plt.savefig(f"{path}{name_plot}.png")

def plot_confusion_matrix(target, predictions, path, name_plot):
    cm = confusion_matrix(target, predictions, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0,1])
    disp.plot()
    plt.savefig(f"{path}{name_plot}.png")