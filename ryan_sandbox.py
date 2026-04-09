# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, random_split, Subset
# import time
# import matplotlib.pyplot as plt
# import numpy as np
# import scripts.file_operations
# import scripts.dataset_utils

# DATA_PATH = '/Volumes/giDrive' #'./data'

# # Configure the device to use GPU (cuda) if available, otherwise MPS (mac) if available, otherwise fallback to CPU device_name = 'cpu'
# device_name = 'cpu' # Fallback to CPU
# if torch.cuda.is_available(): # Prefer CUDA
#     device_name = 'cuda'
# elif torch.backends.mps.is_available(): # Use METAL if CUDA is unavailable
#     device_name = 'mps'

# # Set Device
# device = torch.device(device_name)

# # Print the device being used to verify GPU acceleration
# print(f"Using device: {device}")

# # Delete any lingering MacOS Preview Files (these break the torchvision loaders)
# scripts.file_operations.delete_ds_store(DATA_PATH)

# # Set manual seeds for both PyTorch and NumPy to ensure reproducible results
# torch.manual_seed(42)
# np.random.seed(42)

# # Initialize a dictionary to store and compare results from different experiments
# results = {}

# # Define the data transformations for training: Add RandomHorizontalFlip for augmentation
# # train_transform = transforms.Compose([
# #     transforms.RandomHorizontalFlip(), # Randomly flip images to help the model generalize
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.5,), (0.5,))
# # ])

# # # Define the data transformations for testing: No augmentation needed
# # test_transform = transforms.Compose([
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.5,), (0.5,))
# # ])

# # transfer_transform = transforms.Compose([
# #     transforms.Resize(224),                   # Resize to ImageNet standard
# #     transforms.Grayscale(num_output_channels=3), # Convert to 3-channel RGB
# #     transforms.RandomHorizontalFlip(),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
# # ])

# transfer_transform = transforms.Compose([
#     transforms.Resize(256),                   # Resize shortest edge to 256
#     transforms.CenterCrop(224),               # Crop the center to exactly 224x224
#     # transforms.Grayscale(num_output_channels=3), 
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
# ])

# # Download and load the full training dataset (60,000 images)
# full_dataset = torchvision.datasets.INaturalist(root=DATA_PATH,
#                                              version='2021_train_mini',
#                                              target_type="full",
#                                              transform = transfer_transform,
#                                              download = False)

# """
# Data is stored in directories. Each directory is named with the category attribute
# Data Category Structure:

# 00000_Animalia_Annelida_Clitellata_Haplotaxida_Lumbricidae_Lumbricus_terrestris

# 00000 = Category ID (Numeric)
# Animalia — Kingdom
# Annelida — Phylum
# Clitellata — Class
# Haplotaxida — Order
# Lumbricidae — Family
# Lumbricus — Genus
# terrestris — Species

# """

# # Subset the dataset to only include plants
# plant_dataset = scripts.dataset_utils.return_specified_kingdom(full_dataset=full_dataset, kingom_name="Plantae")

# # train_size = int(0.8 * len(plant_dataset))
# # test_size = len(plant_dataset) - train_size
# # train_set, test_set = random_split(plant_dataset, [train_size, test_size])

# # Flatten nested subsets and create contiguous integer labels
# flat_dataset = scripts.dataset_utils.FlatDataset(plant_dataset)
# num_plant_classes = flat_dataset.num_classes
# print(f"Num Classes: {num_plant_classes}")

# # smaller datasets to start
# total_len = len(flat_dataset)
# train_size_mini = int(0.4 * total_len)
# test_size_mini = int(0.1 * total_len)
# unused_size = total_len - train_size_mini - test_size_mini

# train_set, test_set, _ = random_split(flat_dataset, [train_size_mini, test_size_mini, unused_size])

# # Create DataLoaders for efficient batch processing using the whole dataset
# # train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
# # # Test loader uses the test set for final evaluation
# # test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
# # Print dataset sizes to verify loading
# print(f"Dataset initialization complete. Train: {len(train_set)}, Test: {len(test_set)}")
# print(f"Images resized to 224x224x3")

# transfer_train_loader = DataLoader(train_set, batch_size=64, shuffle=True) # Smaller batch size due to larger images
# transfer_test_loader = DataLoader(test_set, batch_size=64, shuffle=False)



# # 1. Get the integer ID for the "Plantae" kingdom
# plantae_id = full_dataset.categories_index['kingdom']['Plantae']

# # 2. Extract all unique category IDs that belong to Plantae
# plant_class_ids = [
#     cat_id for cat_id, taxonomy in full_dataset.categories_map.items() 
#     if taxonomy['kingdom'] == plantae_id
# ]

# # 3. Get the total count
# num_plant_classes = len(plant_class_ids)

# print(f"There are {num_plant_classes} plant classifications.")



# # TRAIN MODEL

# def train_model(model, train_loader, test_loader, epochs=5, lr=0.01, name="Model"):
#     """
#     Generic training loop with validation. Returns all metrics for comparison.
#     """
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9) 
    
#     print(f"\nTraining {name} for {epochs} epochs...")
#     start_time = time.time()
    
#     # Metrics to track
#     history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
#     for epoch in range(epochs):
#         # --- Training Phase ---
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         train_loss = running_loss / len(train_loader)
#         train_acc = 100 * correct / total
        
#         # --- Validation Phase ---
#         model.eval()
#         val_loss = 0.0
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
        
#         epoch_val_loss = val_loss / len(test_loader)
#         epoch_val_acc = 100 * correct / total
        
#         print(f"  Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
#         history['train_loss'].append(train_loss)
#         history['train_acc'].append(train_acc)
#         history['val_loss'].append(epoch_val_loss)
#         history['val_acc'].append(epoch_val_acc)
            
#     duration = time.time() - start_time
#     print(f"{name} - Final Accuracy: {history['val_acc'][-1]:.2f}%, Time: {duration:.2f}s")
    
#     return history['train_acc'][-1], history['train_loss'][-1], history['val_acc'][-1], history['val_loss'][-1], duration



# # TRANSFER MODEL

# def get_transfer_model(model_name='resnet18', num_classes=10, feature_extract=True):
#     """
#     A generalized function to perform transfer learning on any torchvision model.
#     1. Loads pre-trained weights.
#     2. Freezes the backbone (optional).
#     3. Replaces the final classifier for the target number of classes.
    
#     NOTE: Since our data pipeline converts grayscale to 3-channel RGB using
#     transforms.Grayscale(num_output_channels=3), we do NOT modify the input layer.
#     The pre-trained weights work perfectly with 3-channel input.
#     """
#     # 1. Dynamically load the model from torchvision.models
#     if hasattr(models, model_name):
#         model_func = getattr(models, model_name)
#         # IMPORTANT: For GoogLeNet, we must set transform_input=False 
#         # because it expects 3-channel ImageNet normalization by default.
#         if model_name == 'googlenet':
#             model = model_func(weights='DEFAULT', transform_input=False)
#         else:
#             model = model_func(weights='DEFAULT')
#     else:
#         raise ValueError(f"Model {model_name} not found in torchvision.models")
    
#     # 2. Freeze all parameters initially
#     for param in model.parameters():
#         param.requires_grad = False
        
#     # 3. Keep the first convolutional layer unchanged
#     # Since our data pipeline converts grayscale images to 3-channel RGB,
#     # the pre-trained first layer (which expects 3 channels) works perfectly.
#     # This preserves all the pre-trained knowledge from ImageNet without any adaptation.
    
#     # 4. Replace the final classifier (the 'head')
#     if hasattr(model, 'fc'): # ResNet, GoogLeNet
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Linear(num_ftrs, num_classes)
#         model.fc.weight.requires_grad = True
#         model.fc.bias.requires_grad = True
#     elif hasattr(model, 'classifier'): # VGG
#         if isinstance(model.classifier, nn.Sequential):
#             num_ftrs = model.classifier[-1].in_features
#             model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
#             model.classifier[-1].weight.requires_grad = True
#             model.classifier[-1].bias.requires_grad = True
#         else:
#             num_ftrs = model.classifier.in_features
#             model.classifier = nn.Linear(num_ftrs, num_classes)
#             model.classifier.weight.requires_grad = True
#             model.classifier.bias.requires_grad = True
            
#     # 5. If not feature extracting, unfreeze the whole model (Fine-tuning)
#     if not feature_extract:
#         for param in model.parameters():
#             param.requires_grad = True
            
#     return model

# # ------------------------------------------------------------------------



# # Evaluation 4: Transfer Learning with Fine-Tuning (ResNet-18)
# # For better accuracy, we use feature_extract=False to perform fine-tuning
# model = get_transfer_model('resnet18', num_classes=num_plant_classes, feature_extract=False)

# # Count trainable parameters
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

# # Train the model with a smaller learning rate for fine-tuning
# # We also use a slightly larger number of epochs if needed, but 5 is a good start.
# train_acc, train_loss, val_acc, val_loss, t = train_model(
#     model, 
#     transfer_train_loader, 
#     transfer_test_loader, 
#     epochs=5, 
#     lr=0.001, # Smaller LR for fine-tuning to preserve pre-trained features
#     name="Transfer-ResNet18"
# )

# # Save results
# results['Transfer-ResNet18'] = {
#     'train_acc': train_acc, 
#     'train_loss': train_loss,
#     'val_acc': val_acc, 
#     'val_loss': val_loss,
#     'params': total_params, 
#     'time': t
# }

# # ------------------------------------------------------------------------





# """
# Outstanding:
# 1) Figure out how to isolate data to just vermont (Giles)
# 2) Make sure this data is the right input format for the model (what resoluion are the imaes? need downscaling?)
# 3) Add code to train/evaluate/setup model
# 4) Assess performance of normal model (not finetuned) on test set

# Look into fine tuning (ryan) Do we need to do that?

# 5) Fine-Tune model on training set & re-assess performance

# Misc:
# 1) Some stats of data (how many species, how many data points, etc). maybe PCA/other visualizations
# 2) 
# """

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

"""
This script loads the dataset, filters it down to just vermont, gets some statistics on that dataset
and then saves a report to the /graphs_and_stats directory
"""

# ------------ Initial Setup ------------
# Pick a dataset directory dynamically to load the data
CURRENT_DATASET_NAME = c.MINI_DATASET
local_directory_path = Path(c.MINI_LOCAL_DATA_DIR, CURRENT_DATASET_NAME)
external_directory_path = Path(c.EXTERNAL_DATA_DIR, CURRENT_DATASET_NAME)
system_directory_path = Path(c.SYSTEM_DATA_DIR, CURRENT_DATASET_NAME)

DATA_PATH = ""
if local_directory_path.is_dir():
    DATA_PATH = c.MINI_LOCAL_DATA_DIR
    print(f"Loading Dataset {CURRENT_DATASET_NAME} from path {local_directory_path}")
elif system_directory_path.is_dir():
    DATA_PATH = c.SYSTEM_DATA_DIR
    print(f"Loading Dataset {CURRENT_DATASET_NAME} from path {system_directory_path}")
elif external_directory_path.is_dir():
    DATA_PATH = c.EXTERNAL_DATA_DIR
    print(f"Loading Dataset {CURRENT_DATASET_NAME} from path {external_directory_path}")
else:
    print(f"ERROR - data for dataset {CURRENT_DATASET_NAME} not found in directory "
          f"{local_directory_path} OR {system_directory_path} OR {external_directory_path} "
          f"\n Check Paths & Dataset Name")

# ------------ Initial Setup ------------
# CURRENT_DATASET_NAME = c.FULL_DATASET
# DATA_PATH = str(Path(__file__).resolve().parent.parent.parent / "data")
REPORT_DIRECTORY = str(Path(__file__).resolve().parent.parent / "graphs_and_stats")
TOO_FEW_THRESHOLD = 10
# ------------ Load Data ------------
print("------ Begin Loading Data ------")

# Delete any lingering MacOS Preview Files (these break the torchvision loaders)
scripts.file_operations.delete_ds_store(DATA_PATH)

# Download and load the full training dataset
full_dataset = torchvision.datasets.INaturalist(root=DATA_PATH,
                                             version=CURRENT_DATASET_NAME,
                                             target_type="full",
                                             download = False)

# Subset the dataset to only include plants
plant_dataset = scripts.dataset_utils.return_specified_kingdom(full_dataset=full_dataset, kingom_name="Plantae")

# Subset the dataset further to only include Vermont images
vermont_plant_dataset = scripts.dataset_utils.return_vermont_images(plant_dataset, dataset_name=CURRENT_DATASET_NAME)

# Flatten nested subsets and create contiguous integer labels
flat_dataset = scripts.dataset_utils.FlatDataset(vermont_plant_dataset)

# ------------ Get Statistics on Data ------------
num_plant_classes = int(flat_dataset.num_classes)
total_num_samples = 0
sample_counts = []

# Build a mapping: label -> list of dataset indices
class_to_indices = defaultdict(list)
for idx in range(len(flat_dataset)):
    real_idx = flat_dataset.indices[idx]
    cat_id, _ = flat_dataset.base_dataset.index[real_idx]
    label = flat_dataset.cat_id_to_label[cat_id]
    class_to_indices[label].append(idx)

print("----- Class Encoding & Num Samples-----")
# Now iterate class by class
for label, indices in class_to_indices.items():
    class_name = flat_dataset.label_to_category[label]
    num_samples_in_class = int(len(indices))
    print(f"Class {label} ({class_name}): {num_samples_in_class} samples")
    total_num_samples = total_num_samples + num_samples_in_class
    sample_counts.append(num_samples_in_class)

report_path = os.path.join(REPORT_DIRECTORY, f"{CURRENT_DATASET_NAME}_dataset_report.txt")

# Calculate Additional Statistics
num_below_threshold = 0
for species_count in sample_counts:
    if species_count < TOO_FEW_THRESHOLD:
        num_below_threshold = num_below_threshold + 1

# Build & Save Final Report
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"----- {CURRENT_DATASET_NAME} Dataset Report -----\n")
    f.write("=================================\n\n")

    f.write("-------- General Metrics --------\n")
    f.write(f"Number of Plant Classes: {int(num_plant_classes)}\n")
    f.write(f"Average Samples Per Class: {(total_num_samples/num_plant_classes):.1f}\n")
    f.write(f"Median Samples Per Class: {int(statistics.median(sample_counts))}\n")
    f.write(f"Min Samples Per Class: {int(min(sample_counts))}\n")
    f.write(f"Max Samples Per Class: {int(max(sample_counts))}\n")
    f.write(f"Classes with fewer than {TOO_FEW_THRESHOLD} examples: {int(num_below_threshold)} out of {int(num_plant_classes)}\n")
    f.write("\n")

print(f"Saved model report to: {report_path}")

# ------------ Visualizations ------------
# Get per-class counts sorted in descending order
sorted_counts = sorted([len(indices) for indices in class_to_indices.values()], reverse=True)

# --- Sorted Bar Chart ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(sorted_counts)), sorted_counts, color="steelblue", edgecolor="none")
ax.set_xlabel("Species (sorted by sample count)")
ax.set_ylabel("Number of Samples")
ax.set_title(f"{CURRENT_DATASET_NAME} — Class Distribution (sorted)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIRECTORY, f"{CURRENT_DATASET_NAME}_class_distribution.png"), dpi=150)
plt.show()
