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
max_pats = []
mean_pats = []
quantil25 = []
quantil50 = []
quantil75 = []
restes = []
eps=10

for dir in directories:
    if metadata.loc[metadata["CODI"] == dir.split("/")[-1].split("_")[0], "DENSITAT"].values[0] == "NEGATIVA":
        target = 0
    else:
        target = 1
    files = os.listdir(dir)
    if len(files) == 0:
        continue
    predict_patches = 0
    prop_pac = []
    resta = []
    for file in files:
        img = read_image(os.path.join(dir, file))[:-1,:,:]
        img = img.to(torch.float32)
        img = img/255
        img = transform(img)
        model.eval()
        img_processed = model(img, "autoencoder")
        red_pixels_original = red_pixels(img)
        red_pixels_output = red_pixels(img_processed)

        if ((red_pixels_original+eps)/(red_pixels_output+eps)) > 1:
                predict_patches += 1

        if red_pixels_original == 0 and red_pixels_output == 0:
            prop_pac.append(0)
        elif red_pixels_original == 0:
            prop_pac.append(0) 
        else: 
            prop_pac.append(red_pixels_output/red_pixels_original)
        
        resta.append(-red_pixels_output+red_pixels_original)
    prop_pac = sorted(prop_pac)
    max_pat = max(prop_pac)   
    mean_pat = sum(prop_pac)/len(prop_pac)  
    quantil25.append(prop_pac[int(len(prop_pac)/10)*7]) 
    quantil50.append(prop_pac[int(len(prop_pac)/10)*8]) 
    quantil75.append(prop_pac[int(len(prop_pac)/10)*9])
    restes.append(max(resta))
    prop = predict_patches/len(files)

    targets.append(target)
    props.append(prop)
    max_pats.append(max_pat)
    mean_pats.append(mean_pat)
df = pd.DataFrame({'max': max_pats, 'mean': mean_pats, 'prop': props, 'q25': quantil25, 'q50': quantil50, 'q75': quantil75, 'resta': restes, 'target': targets})
print(df)

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


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the breast cancer dataset for binary classification
X = df.drop(columns = ["target"])
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a GradientBoostingClassifier for binary classification
gb_classifier = GradientBoostingClassifier(
    n_estimators=100,           # Number of boosting rounds (trees to build)
    learning_rate=0.1,          # Step size shrinkage to prevent overfitting
    max_depth=3,                # Maximum depth of a tree
    random_state=42             # Seed for reproducibility
)

# Train the classifier on the training data
gb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = gb_classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)
        




    
