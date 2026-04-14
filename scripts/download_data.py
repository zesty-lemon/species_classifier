import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import dataset_utils.file_operations
from utils import dataset_utils

"""
Downloading the data takes a long time
This standalone script will pull from torchvision
You can run it in the background (from terminal) while doing other work

"""
# ------------ Initial Setup ------------
# DATA_PATH = '/Volumes/giDrive' # External Volume
DATA_PATH = '../data'  # Local Storage
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

# Initialize a dictionary to store and compare results from different experiments
results = {}

# Set manual seeds for both PyTorch and NumPy to ensure reproducible results
torch.manual_seed(42)
np.random.seed(42)

# ------------ Load Data ------------
print("------ BEGIN Loading Data ------")
# Define the data transformations for testing: No augmentation needed
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Transform images to resnet standard
transfer_transform = transforms.Compose([
    transforms.Resize((224,224)), # Resize to ImageNet standard
    transforms.Grayscale(num_output_channels=3), # Convert to 3-channel RGB
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

# Delete any lingering MacOS Preview Files (these break the torchvision loaders)
dataset_utils.delete_ds_store(DATA_PATH)

# Download and load the full training dataset
full_dataset = torchvision.datasets.INaturalist(root=DATA_PATH,
                                             version='2021_valid',
                                             target_type="full",
                                             transform = transfer_transform,
                                             download = True)

# Print dataset sizes to verify loading
print("------ END Loading Data ------")