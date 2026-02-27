import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import time
import matplotlib.pyplot as plt
import numpy as np
import scripts.file_operations
import scripts.dataset_utils

# ------------ Initial Setup ------------
DATA_PATH = '/Volumes/giDrive'
DATA_PATH = './data'
# Configure the device to use GPU (cuda) if available, otherwise MPS (mac) if available, otherwise fallback to CPU device_name = 'cpu'
device_name = 'cpu' # Fallback to CPU
if torch.cuda.is_available(): # Prefer CUDA
    device_name = 'cuda'
elif torch.backends.mps.is_available(): # Use METAL if CUDA is unavailable
    device_name = 'mps'

# Set Device
device = torch.device(device_name)

# Print the device being used to verify GPU acceleration
print(f"Using device: {device}")

# Set manual seeds for both PyTorch and NumPy to ensure reproducible results
torch.manual_seed(42)
np.random.seed(42)

# ------------ Load Data ------------
# Define the data transformations for training: Add RandomHorizontalFlip for augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # Randomly flip images to help the model generalize
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the data transformations for testing: No augmentation needed
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Delete any lingering MacOS Preview Files (these break the torchvision loaders)
scripts.file_operations.delete_ds_store(DATA_PATH)

# Download and load the full training dataset
full_dataset = torchvision.datasets.INaturalist(root=DATA_PATH,
                                             version='2021_train_mini',
                                             target_type="full",
                                             transform = train_transform,
                                             download = False)


# Subset the dataset to only include plants
plant_dataset = scripts.dataset_utils.return_specified_kingdom(full_dataset=full_dataset, kingom_name="Plantae")

# # Subset the dataset further to only include Vermont images
# vermont_dataset = scripts.dataset_utils.return_vermont_images(plant_dataset)

train_size = int(0.8 * len(plant_dataset))
test_size = len(plant_dataset) - train_size
train_set, test_set = random_split(plant_dataset, [train_size, test_size])

# Create DataLoaders for efficient batch processing using the whole dataset
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
# Test loader uses the test set for final evaluation
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
# Print dataset sizes to verify loading
print(f"Dataset initialization complete. Train: {len(train_set)}, Test: {len(test_set)}")

# ------------ Train Model ------------





"""
Outstanding:
1) Figure out how to isolate data to just vermont (Giles)
2) Make sure this data is the right input format for the model (what resoluion are the imaes? need downscaling?)
3) Add code to train/evaluate/setup model
4) Assess performance of normal model (not finetuned) on test set

Look into fine tuning (ryan) Do we need to do that?

5) Fine-Tune model on training set & re-assess performance

Misc:
1) Some stats of data (how many species, how many data points, etc). maybe PCA/other visualizations
2) 
"""