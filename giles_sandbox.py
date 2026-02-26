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

# Initialize a dictionary to store and compare results from different experiments
results = {}

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

# Download and load the full training dataset (60,000 images)
full_dataset = torchvision.datasets.INaturalist(root='./data',
                                             version='2021_train_mini',
                                             target_type="full",
                                             transform = train_transform,
                                             download = True)
"""
Data Categories: Data has a category attribute
00000_Animalia_Annelida_Clitellata_Haplotaxida_Lumbricidae_Lumbricus_terrestris

00000 = Category ID (Numeric)
Animalia — Kingdom
Annelida — Phylum
Clitellata — Class
Haplotaxida — Order
Lumbricidae — Family
Lumbricus — Genus
terrestris — Species

The category corresponds to a directory inside of which are images of that particular species
"""

# Find category IDs where kingdom is "Plantae"
plantae_cat_ids = set()
for index in range(len(full_dataset.all_categories)):
    category = full_dataset.all_categories[index]

    if'Plantae' in category:
        plantae_cat_ids.add(index)

# Find dataset indices (list) that belong to those categories
plantae_indices = [
    i for i, (cat_id, _) in enumerate(full_dataset.index)
    if cat_id in plantae_cat_ids
]

# Create the filtered subset
plantae_dataset = Subset(full_dataset, plantae_indices)

train_size = int(0.8 * len(plantae_dataset))
test_size = len(plantae_dataset) - train_size
train_set, test_set = random_split(plantae_dataset, [train_size, test_size])

# Create DataLoaders for efficient batch processing using the whole dataset
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
# Test loader uses the test set for final evaluation
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
# Print dataset sizes to verify loading
print(f"Dataset initialization complete. Train: {len(train_set)}, Test: {len(test_set)}")

"""
Todo:

1. Download the data
2. Split out just vermont
3. Split out just plants ("plantae")
4. PCA - Giles

"""