from model_autoencoder import Autoencoder
from dataset import Dataset
import torch
import os
import pandas as pd
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import read_image
from config import SAVED_MODEL, ANNOTATED_PATCHES_DIR, METADATA_FILE, WINDOW_METADATA_FILE, PLOT_LOSS_DIR

model = Autoencoder()
model.load_state_dict(torch.load(SAVED_MODEL))

metadata = pd.read_csv(METADATA_FILE)
window_metadata = pd.read_csv(WINDOW_METADATA_FILE)

directories = [dir.path for dir in os.scandir(ANNOTATED_PATCHES_DIR) if dir.is_dir()]

transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
    transforms.Normalize(mean=[0.5], std=[0.5])])

def red_pixels(img):
    img = img.permute(1, 2, 0).detach().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_limit = 340
    upper_limit = 20
    count = ((img[:, :, 0] >= lower_limit) | (img[:, :, 0] <= upper_limit)).sum()

    return count

targets = []
props = []
target_patch = []
val_ppatch = []
prop_pat = []
eps = 10
dirs_patch = []

for dir in directories:
    if metadata.loc[metadata["CODI"] == dir.split("/")[-1].split("_")[0], "DENSITAT"].values[0] == "NEGATIVA":
        target = 0
    else:
        target = 1
    files = os.listdir(dir)
    if len(files) == 0:
        continue
    predict_patches = 0
    for file in files:
        if window_metadata.loc[window_metadata["ID"] == str(dir.split("/")[-1])+"."+file[:-4], "Presence"].values == -1:
            target_patch.append(0)
        else:
            target_patch.append(1)
        img = read_image(os.path.join(dir, file))[:-1,:,:]
        img = img.to(torch.float32)
        img = img/255
        img = transform(img)
        model.eval()
        img_processed = model(img, "autoencoder")
        red_pixels_original = red_pixels(img)
        red_pixels_output = red_pixels(img_processed)

        if red_pixels_original > red_pixels_output:
                predict_patches += 1
                val_ppatch.append(1)
        else:
            val_ppatch.append(0)
        prop_pat.append((red_pixels_original+eps)/(red_pixels_output+eps))
        dirs_patch.append(dir+'/'+file)
    prop = predict_patches/len(files)

    targets.append(target)
    props.append(prop)

final_results = pd.DataFrame({"Target": targets, "Prop": props})

fpr, tpr, thresholds = roc_curve(final_results['Target'], final_results['Prop'])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig(f"{PLOT_LOSS_DIR}ROCcurve.png")

J = tpr - fpr
best_threshold = thresholds[np.argmax(J)]
print(best_threshold)

# Define the target and predicted labels using the best threshold
predicted_labels = [0 if prob < best_threshold else 1 for prob in props]

# Calculate the confusion matrix
confusion_matrix = metrics.confusion_matrix(targets, predicted_labels)

# Print the confusion matrix
print("Confusion matrix:")
print(confusion_matrix)

final_results2 = pd.DataFrame({"Target": target_patch, "Prop": prop_pat, "Dirs": dirs_patch})

fpr2, tpr2, thresholds2 = roc_curve(final_results2['Target'], final_results2['Prop'])
roc_auc2 = auc(fpr2, tpr2)
plt.figure(figsize=(8, 6))
plt.plot(fpr2, tpr2, color='darkorange', lw=2, label=f'ROC curve patches (AUC = {roc_auc2:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig(f"{PLOT_LOSS_DIR}ROCcurve2.png")

J2 = tpr2 - fpr2
best_threshold2 = thresholds2[np.argmax(J2)]
print(best_threshold2)

# Define the target and predicted labels using the best threshold
predicted_labels2 = [0 if prob2 < best_threshold2 else 1 for prob2 in prop_pat]


# Calculate the confusion matrix
confusion_matrix2 = metrics.confusion_matrix(target_patch, predicted_labels2)

# Print the confusion matrix
print("Confusion matrix patches:")
print(confusion_matrix2)


        

print(final_results2)


    
