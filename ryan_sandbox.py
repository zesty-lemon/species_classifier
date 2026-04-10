#     running_loss += loss.item()

#     # _, predicted = torch.max(outputs.data, 1)
#     # correct += (predicted == labels).sum().item()

#     # top k outputs instead of 1
#     _, predicted = torch.topk(outputs.data, 5, dim=1)
#     # correct if predicted is in top k outputs
#     correct_top5 += (predicted == labels.view(-1, 1)).sum().item()

#     total += labels.size(0)

# train_loss = running_loss / len(train_loader)
# # train_acc = 100 * correct / total
# train_acc_top5 = 100 * correct_top5 / total

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

"""
This script loads the dataset, filters it down to just vermont, gets some statistics on that dataset
and then saves a report to the /graphs_and_stats directory
"""
# ------------ Initial Setup ------------
# CURRENT_DATASET_NAME = c.MINI_DATASET
# DATA_PATH = str(Path(__file__).resolve().parent.parent / "data")
REPORT_DIRECTORY = str(Path(__file__).resolve().parent.parent / "graphs_and_stats")

TOO_FEW_THRESHOLD = 10

# ------------ Initial Setup ------------
# Pick a dataset directory dynamically to load the data
CURRENT_DATASET_NAME = c.FULL_DATASET
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
    





# # ------------ Load Data ------------
# print("------ Begin Loading Data ------")

# # Delete any lingering MacOS Preview Files (these break the torchvision loaders)
# scripts.file_operations.delete_ds_store(DATA_PATH)

# # Download and load the full training dataset
# full_dataset = torchvision.datasets.INaturalist(root=DATA_PATH,
#                                              version=CURRENT_DATASET_NAME,
#                                              target_type="full",
#                                              download = False)

# # Subset the dataset further to only include Plants found in Vermont
# vermont_plant_dataset = scripts.dataset_utils.return_species_relevant_to_vermont(dataset=full_dataset, kingom_name="Plantae")

# # Flatten nested subsets and create contiguous integer labels
# flat_dataset = scripts.dataset_utils.FlatDataset(vermont_plant_dataset)


# # ------------ Get Statistics on Data ------------
# num_plant_classes = int(flat_dataset.num_classes)
# total_num_samples = 0
# sample_counts = []

# # Build a mapping: label -> list of dataset indices
# class_to_indices = defaultdict(list)
# for idx in range(len(flat_dataset)):
#     real_idx = flat_dataset.indices[idx]
#     cat_id, _ = flat_dataset.base_dataset.index[real_idx]
#     label = flat_dataset.cat_id_to_label[cat_id]
#     class_to_indices[label].append(idx)







# print("----- Class Encoding & Num Samples-----")
# # Now iterate class by class
# for label, indices in class_to_indices.items():
#     class_name = flat_dataset.label_to_category[label]
#     num_samples_in_class = int(len(indices))
#     print(f"Class {label} ({class_name}): {num_samples_in_class} samples")
#     total_num_samples = total_num_samples + num_samples_in_class
#     sample_counts.append(num_samples_in_class)

# report_path = os.path.join(REPORT_DIRECTORY, f"{CURRENT_DATASET_NAME}_dataset_report.txt")

# # Calculate Additional Statistics
# num_below_threshold = 0
# for species_count in sample_counts:
#     if species_count < TOO_FEW_THRESHOLD:
#         num_below_threshold = num_below_threshold + 1

# # Build & Save Final Report
# with open(report_path, "w", encoding="utf-8") as f:
#     f.write(f"----- {CURRENT_DATASET_NAME} Dataset Report -----\n")
#     f.write("=================================\n\n")

#     f.write("-------- General Metrics --------\n")
#     f.write(f"Number of Plant Classes: {int(num_plant_classes)}\n")
#     f.write(f"Average Samples Per Class: {(total_num_samples/num_plant_classes):.1f}\n")
#     f.write(f"Median Samples Per Class: {int(statistics.median(sample_counts))}\n")
#     f.write(f"Min Samples Per Class: {int(min(sample_counts))}\n")
#     f.write(f"Max Samples Per Class: {int(max(sample_counts))}\n")
#     f.write(f"Classes with fewer than {TOO_FEW_THRESHOLD} examples: {int(num_below_threshold)} out of {int(num_plant_classes)}\n")
#     f.write("\n")

# print(f"Saved model report to: {report_path}")

# # ------------ Visualizations ------------
# # Get per-class counts sorted in descending order
# sorted_counts = sorted([len(indices) for indices in class_to_indices.values()], reverse=True)

# # --- Sorted Bar Chart ---
# fig, ax = plt.subplots(figsize=(14, 5))
# ax.bar(range(len(sorted_counts)), sorted_counts, color="steelblue", edgecolor="none")
# ax.set_xlabel("Species (sorted by sample count)")
# ax.set_ylabel("Number of Samples")
# ax.set_title(f"{CURRENT_DATASET_NAME} — Class Distribution (sorted)")
# ax.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(REPORT_DIRECTORY, f"{CURRENT_DATASET_NAME}_class_distribution.png"), dpi=150)
# plt.show()








# # ------------ Export Filtered Data ------------
# print("\n------ Begin Exporting Filtered Dataset ------")

# # Define where you want to save the new filtered dataset
# # This creates a 'vermont_plants_dataset' folder in your primary DATA_PATH
# EXPORT_DIR = Path(DATA_PATH) / "vermont_plants_dataset"
# EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# # Iterate class by class to save images
# for label, indices in class_to_indices.items():
#     original_class_name = flat_dataset.label_to_category[label]
    
#     # Split the original name at the first underscore to remove the old number
#     # "05729_Plantae..." becomes ["05729", "Plantae..."]
#     parts = original_class_name.split("_", 1)
    
#     # Prepend the new ascending label (zero-padded to 5 digits)
#     if len(parts) == 2:
#         new_class_name = f"{label:05d}_{parts[1]}"
#     else:
#         new_class_name = f"{label:05d}_{original_class_name}"
        
#     # Create the directory for this specific class
#     class_dir = EXPORT_DIR / new_class_name
#     class_dir.mkdir(parents=True, exist_ok=True)
    
#     print(f"Exporting Class {label}: {new_class_name} ({len(indices)} images)...")
    
#     # Save each image in the class
#     for i, dataset_idx in enumerate(indices):
#         # Retrieve the image from the dataset. 
#         # By default, torchvision datasets return (PIL_Image, label) 
#         # when no transforms are applied.
#         image, _ = flat_dataset[dataset_idx]
        
#         # Create a unique filename for the image
#         file_name = f"{new_class_name}_img_{i:04d}.jpg"
#         file_path = class_dir / file_name
        
#         # Save the PIL image to disk
#         image.save(file_path)

# print(f"\nFiltered dataset exported to: {EXPORT_DIR}")







# ------------ Access Dowloaded Vermont Dataset Without Augmentation ------------
dataset_dir = Path(DATA_PATH) / "vermont_plants_dataset" # "vermont_plants_dataset_mini"

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Standardizes image sizes
    transforms.ToTensor()          # Converts PIL Image to PyTorch Tensor
])

# Load the dataset using ImageFolder
vermont_dataset = ImageFolder(root=str(dataset_dir), transform=img_transforms)

vermont_dataloader = DataLoader(vermont_dataset, batch_size=32, shuffle=True, num_workers=4)

# --- Verification ---
print(f"Successfully loaded {len(vermont_dataset)} images.")
print(f"Number of classes: {len(vermont_dataset.classes)}")









# # ------------ Image Augmentation ------------
# train_transforms = transforms.Compose([
#     # Randomly crops a piece of the image and resizes it to 224x224
#     transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), 
    
#     # 50% chance to flip the image left-to-right (plants aren't direction-dependent)
#     transforms.RandomHorizontalFlip(p=0.5),
    
#     # Randomly rotates the image up to 30 degrees in either direction
#     transforms.RandomRotation(degrees=30),
    
#     # Simulates different lighting, shadows, and camera sensors
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    
#     # Converts the PIL Image to a PyTorch Tensor
#     transforms.ToTensor(),
    
#     # Standardizes pixel values (Crucial if using pre-trained models like ResNet)
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Validation Transform have NO AUGMENTATION
# val_transforms = transforms.Compose([
#     transforms.Resize(256),        # Resize slightly larger first
#     transforms.CenterCrop(224),    # Crop the exact center
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # ------------ Load Downloaded Vermont Dataset with Augmentations ------------
# base_dataset = ImageFolder(root=dataset_dir)


# # Split the dataset (e.g., 80% train, 20% validation)
# train_size = int(0.8 * len(base_dataset))
# val_size = len(base_dataset) - train_size
# train_subset, val_subset = random_split(base_dataset, [train_size, val_size])

# # We need a custom wrapper to apply different transforms to each subset
# class DatasetWrapper:
#     def __init__(self, subset, transform=None):
#         self.subset = subset
#         self.transform = transform
        
#     def __getitem__(self, index):
#         image, label = self.subset[index]
#         if self.transform:
#             image = self.transform(image)
#         return image, label
        
#     def __len__(self):
#         return len(self.subset)

# # Wrap the subsets with their respective transforms
# train_dataset = DatasetWrapper(train_subset, transform=train_transforms)
# val_dataset = DatasetWrapper(val_subset, transform=val_transforms)

# # Create the DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)