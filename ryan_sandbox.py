import os
import statistics
from pathlib import Path
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import scripts.file_operations
import scripts.dataset_utils
import constants as c
from collections import defaultdict
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import copy

scripts.dataset_utils.download_vermont_images(dataset_name='2021_train_mini', data_filepath='/Volumes/giDrive/data')

# # ------------ Access Dowloaded Vermont Dataset Without Augmentation ------------
# dataset_dir = Path(DATA_PATH) / "vermont_plants_dataset" # "vermont_plants_dataset_mini"

# img_transforms = transforms.Compose([
#     transforms.Resize((224, 224)), # Standardizes image sizes
#     transforms.ToTensor()          # Converts PIL Image to PyTorch Tensor
# ])

# # Load the dataset using ImageFolder
# vermont_dataset = ImageFolder(root=str(dataset_dir), transform=img_transforms)

# vermont_dataloader = DataLoader(vermont_dataset, batch_size=32, shuffle=True, num_workers=4)

# # --- Verification ---
# print(f"Successfully loaded {len(vermont_dataset)} images.")
# print(f"Number of classes: {len(vermont_dataset.classes)}")