from model_autoencoder import Autoencoder
from dataset_autoencoder import Dataset
import torch
import os
import pandas as pd
import cv2
import numpy as np
from torchvision import transforms
from torchvision.io import read_image
from config import SAVED_MODEL, ANNOTATED_PATCHES_DIR, METADATA_FILE, WINDOW_METADATA_FILE

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
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = torch.from_numpy(img)
    lower_limit = 300
    upper_limit = 60
    mask = (img[:, :, 0] >= lower_limit) | (img[:, :, 0] <= upper_limit)
    count = torch.sum(mask)

    return count

targets = []
props = []

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
        img = read_image(os.path.join(dir, file))
        img = img.to(torch.float32)
        img = img/255
        img = transform(img)
        model.eval()
        img_processed = model(img, "mlp")
        red_pixels_original = red_pixels(img)
        red_pixels_output = red_pixels(img_processed)

        if red_pixels_original >= red_pixels_output:
            predict_patches += 1
            
    prop = predict_patches/len(files)

    targets.append(target)
    props.append(prop)

final_results = pd.DataFrame({"Target": targets, "Prop": prop})
print("d")
        




    
