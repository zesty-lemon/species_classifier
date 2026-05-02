"""
VLM Rescue Experiment

We use resnet50 to get the classification, but if it is not confident we use a VLM layer to fine-tune the
prediction

We go through all images in valdiation set
We take an image, run it through the model, and get back the top 5 predicted classes
If the margin (the difference in confidence between the top 2 classes) is low,
the model is not confident in its classification. If the model is not confident,
we send the image and the top 5 class labels to the VLM and let the VLM decide which should be the top class.
"""
import sys

import anthropic
import torch
from PIL import Image
from scipy.stats import binomtest
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv() #loads the api key from .env file
from config import constants as c
import base64
import os

# janky fix
# unpickling the models introduces errors since directory structure changed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from config.device_config import device, device_name
import utils.data_load_and_config_util as data_config
from config import constants as c
from models.model_utils.model_utils import get_cuda_trained_model
from utils.data_load_and_config_util import get_test_transfer_transforms

def clear():
    # 'nt' is for Windows, 'posix' is for Linux/macOS
    os.system('cls' if os.name == 'nt' else 'clear')

# ------------ Initial Configuration ------------
MODEL_NAME = "ResNet101 Scratch Trained"
MODEL_PATH = str(c.PROJECT_ROOT / "models" / "trained_models" / "resnet_101" / "ResNet101_Scratch_Trained_model.joblib")
DATASET_USED = c.FULL_DATASET # Need to leave this even though it isn't being used, need it for category indexes
TOP_K = 5
MARGIN_THRESHOLD = 0.5  # Only send to VLM when ResNet's margin < this value

# ------------ Dataset Setup ------------
CURRENT_DATASET_NAME, DATA_PATH = data_config.get_dataset_name_and_path(DATASET_USED)

# ------------ Initialize Loaders ------------
_, val_loader, num_plant_classes = data_config.load_vermont_plant_data(dataset_name=CURRENT_DATASET_NAME,
                                                                                  data_path=DATA_PATH,
                                                                                  device_name=device_name,
                                                                                  batch_size=128)

trained_model = get_cuda_trained_model(MODEL_PATH)
trained_model = trained_model.to(device)
trained_model.eval()

label_to_category = val_loader.dataset.label_to_category

flat_dataset = val_loader.dataset
base_dataset = flat_dataset.base_dataset

clear()

# image_path = '/Users/giles/Documents/Grad_School/Spring_2026/deep_learning/Project/data/2021_valid/08207_Plantae_Tracheophyta_Magnoliopsida_Fagales_Fagaceae_Quercus_velutina/d574da4c-7565-4153-a25a-b2c5117807b1.jpg'
print("============ Vermont Species Classifier ============")
print("Model: Resnet101 (Scratch Trained)")
print("Drag in image to be classified")
image_path = input("Drag Image Here: ")
image_path = image_path.replace(" ", "")

image = Image.open(image_path).convert("RGB")

_, val_transform = get_test_transfer_transforms()
input_tensor = val_transform(image).unsqueeze(0).to(device)


# Get the true category for the image
category_to_label = {name: lbl for lbl, name in label_to_category.items()}
true_category = os.path.basename(os.path.dirname(image_path))
label = category_to_label[true_category]

with torch.no_grad():
    logits = trained_model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    topk_probs, topk_idx = probs.topk(TOP_K, dim=1)

    # probabilities come back as tensors, convert to indexes
    topk_probs_list = topk_probs.tolist()[0]
    topk_idx_list = topk_idx.tolist()[0]
    margin = topk_probs_list[0] - topk_probs_list[1]

    baseline_top1_correct = (topk_idx_list[0] == label)
    print(f"True Category: {true_category}")
    print(f"Top-1 prediction: {label_to_category[topk_idx_list[0]]} (correct: {baseline_top1_correct})")
    print("Top 5 Predictions:")
    for i in range (0,len(topk_idx_list)):
        print(f"    Prediction: {label_to_category[topk_idx_list[i]]} (Confidence: {topk_probs_list[i]:.2%})")
