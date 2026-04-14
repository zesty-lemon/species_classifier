import os
import statistics
from pathlib import Path

import torchvision
import matplotlib.pyplot as plt
from config import constants as c
from collections import defaultdict

from utils import dataset_utils

"""
This script loads the dataset, filters it down to just vermont, gets some statistics on that dataset
and then saves a report to the /graphs_and_stats directory
"""
# ------------ Initial Setup ------------
CURRENT_DATASET_NAME = c.VAL_DATASET
DATA_PATH = str(Path(__file__).resolve().parent.parent / "data")
REPORT_DIRECTORY = str(Path(__file__).resolve().parent.parent / "graphs_and_stats")

TOO_FEW_THRESHOLD = 10
# ------------ Load Data ------------
print("------ BEGIN Loading Data ------")

# Delete any lingering MacOS Preview Files (these break the torchvision loaders)
dataset_utils.delete_ds_store(DATA_PATH)

# Download and load the full training dataset
full_dataset = torchvision.datasets.INaturalist(root=DATA_PATH,
                                             version=CURRENT_DATASET_NAME,
                                             target_type="full",
                                             download = False)

# Subset the dataset further to only include Plants found in Vermont
vermont_plant_dataset = dataset_utils.return_species_relevant_to_vermont(dataset=full_dataset, dataset_name=CURRENT_DATASET_NAME, kingom_name="Plantae")

# Flatten nested subsets and create contiguous integer labels
flat_dataset = dataset_utils.FlatDataset(vermont_plant_dataset)

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
