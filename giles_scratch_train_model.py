from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, Subset
import time
import matplotlib.pyplot as plt
import numpy as np
import scripts.file_operations
import scripts.dataset_utils
import constants as c
import preprocessing.transforms as t
from config.device_config import device, device_name
import config.data_config as data_config

# ------------ Initial Setup ------------
# Get Dataset Name & Path
CURRENT_DATASET_NAME, DATA_PATH = data_config.get_dataset_name_path(c.FULL_DATASET)

# Set manual seeds for both PyTorch and NumPy to ensure reproducible results
torch.manual_seed(42)
np.random.seed(42)

# ------------ Load Data ------------
print("------ Begin Loading Data ------")
# Define the data transformations
test_transform, transfer_transform = t.get_test_transfer_transforms()

# Delete any lingering MacOS Preview Files (these break the torchvision loaders)
scripts.file_operations.delete_ds_store(DATA_PATH)

# Download and load the full training dataset
full_dataset = torchvision.datasets.INaturalist(root=DATA_PATH,
                                             version=CURRENT_DATASET_NAME,
                                             target_type="full",
                                             transform = transfer_transform,
                                             download = False)

# Subset the dataset further to only include Plants found in Vermont
vermont_plant_dataset = scripts.dataset_utils.return_species_relevant_to_vermont(dataset=full_dataset, kingom_name="Plantae")

# Flatten nested subsets and create contiguous integer labels
flat_dataset = scripts.dataset_utils.FlatDataset(vermont_plant_dataset)

num_plant_classes = flat_dataset.num_classes

print(f"Num Classes: {num_plant_classes}")

train_size = int(0.8 * len(flat_dataset))
test_size = len(flat_dataset) - train_size
train_set, test_set = random_split(flat_dataset, [train_size, test_size])

# Create DataLoaders for efficient batch processing using the whole dataset
use_cuda = (device_name == 'cuda')

train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                          num_workers=4 if use_cuda else 0,
                          pin_memory=use_cuda)

# Test loader uses the test set for final evaluation
test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                         num_workers=4 if use_cuda else 0,
                         pin_memory=use_cuda)

# Print dataset sizes to verify loading
print(f"Dataset initialization complete. Train: {len(train_set)}, Test: {len(test_set)}")
print("------ End Loading Data ------")

# ------------ Train Model ------------
def train_model(model, train_loader, test_loader, epochs=5, lr=0.01, name="Model"):
    """
    Generic training loop with validation. Returns all metrics for comparison.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    scaler = torch.amp.GradScaler(enabled=(device_name == 'cuda')) # only enable if running on CUDA

    print(f"\nTraining {name} for {epochs} epochs...")
    start_time = time.time()

    # Metrics to track
    history = {'train_loss': [], 'train_acc': [], 'train_top5_acc': [], 'val_loss': [], 'val_acc': [], 'val_top5_acc': []}

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        top5_correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_name):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _, top5_pred = outputs.topk(5, dim=1)
            top5_correct += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_top5_acc = 100 * top5_correct / total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type=device_name):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                _, top5_pred = outputs.topk(5, dim=1)
                top5_correct += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()

        epoch_val_loss = val_loss / len(test_loader)
        epoch_val_acc = 100 * correct / total
        epoch_val_top5_acc = 100 * top5_correct / total

        current_iteration_status = f"  Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Top-5: {train_top5_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | Val Top-5: {epoch_val_top5_acc:.2f}%"

        print(current_iteration_status)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_top5_acc'].append(train_top5_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_top5_acc'].append(epoch_val_top5_acc)

    duration = time.time() - start_time
    print(f"{name} — Final Val Acc: {history['val_acc'][-1]:.2f}%, Val Top-5 Acc: {history['val_top5_acc'][-1]:.2f}%, Time: {duration:.2f}s")
    return history, duration


def plot_training_curves(history, name="Model"):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{name} — Loss per Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Top-5 Accuracy ---
    ax2.plot(epochs, history["train_top5_acc"], "o-", label="Train Top-5 Acc")
    ax2.plot(epochs, history["val_top5_acc"], "s-", label="Val Top-5 Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-5 Accuracy (%)")
    ax2.set_title(f"{name} — Top-5 Accuracy per Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_')}_training_curves.png", dpi=150)
    plt.show()


class BottleNeck_Block(nn.Module):
    """
    Bottleneck Block for ResNet-50.
    Structure: Conv1x1(reduce) -> Conv3x3 -> Conv1x1(expand) -> skip connection
    """
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, expansion=expansion):
        super(BottleNeck_Block, self).__init__()
        # --- Convolution Layers ---
        # Conv1: 1x1
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Conv2: 3x3, with stride parameter
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # Conv3: 1x1, expand to channels * expansion
        self.conv3 = nn.Conv2d(channels, channels * expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * expansion)

        # --- ReLU Layer ---
        self.relu = nn.ReLU(inplace=True)

        # Default: Skip Connection (identity)
        self.shortcut = nn.Identity()

        # 1x1 conv + bn if dimensions change
        # Adjust shortcut if dimensions change
        if stride != 1 or in_channels != channels * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * expansion)
            )

    # Forward Pass
    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet50_Model(nn.Module):
    """
    ResNet-50 with Bottleneck blocks. Configuration: [3, 4, 6, 3]
    """

    def __init__(self, num_classes=10):
        super(ResNet50_Model, self).__init__()

        # ===== 1. Initial Convolution (Stem) =====
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ===== 2. Max Pooling =====
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ===== 3. Residual Stages =====
        # ResNet-50 configuration: [3, 4, 6, 3] blocks per stage
        self.layer1 = self._make_layer(in_channels=64, channels=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(in_channels=256, channels=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(in_channels=512, channels=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(in_channels=1024, channels=512, blocks=3, stride=2)

        # ===== 4. Classification Head =====
        # Global Average Pooling: reduces any spatial size to 1x1
        # This replaces heavy FC layers (like in VGG) and adds translation invariance
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Final classifier: 512 features -> num_classes
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, channels, blocks, stride=1):
        """
        Create a layer with BottleneckBlock_Exercise blocks.
        First block uses stride, rest use stride=1.
        """
        layers = []
        # First block: handles channel change and optional downsampling
        layers.append(BottleNeck_Block(in_channels=in_channels, channels=channels, stride=stride))

        # Remaining blocks: input channels = output channels of previous block (channels * expansion)
        for _ in range(1, blocks):
            layers.append(BottleNeck_Block(in_channels=channels * BottleNeck_Block.expansion,
                                           channels=channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem: Initial feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Four residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling + flatten
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc(x)
        return x


# Initialize
print("------ Begin Training Model ------")
resnet50_exercise = ResNet50_Model(num_classes=num_plant_classes)

# Count parameters and compare with ResNet-18
params_res50 = sum(p.numel() for p in resnet50_exercise.parameters())
print(f"ResNet-50 Parameters: {params_res50:,}")

# Test: ResNet-50 (Scratch Training)
history, duration = train_model(resnet50_exercise, train_loader, test_loader, epochs=10, lr=0.01, name="ResNet-50")

print("------ End Training Model ------")

plot_training_curves(history, name="ResNet50 - Scratch Trained")
