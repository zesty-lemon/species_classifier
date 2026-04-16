from pathlib import Path
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
import constants as c
from torchvision.datasets import ImageFolder

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

# Where downloaded filtered uniform dataset is located
dataset_dir = Path(DATA_PATH) / "vermont_plants_dataset" # "vermont_plants_dataset_mini"

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
print("------ Begin Loading Data ------")


# ------------ Image Augmentation ------------
train_transforms = transforms.Compose([
    # Randomly crops a piece of the image and resizes it to 224x224
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), 
    
    # 50% chance to flip the image left-to-right (plants aren't direction-dependent)
    transforms.RandomHorizontalFlip(p=0.5),
    
    # Randomly rotates the image up to 30 degrees in either direction
    transforms.RandomRotation(degrees=30),
    
    # Simulates different lighting, shadows, and camera sensors
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    
    # Converts the PIL Image to a PyTorch Tensor
    transforms.ToTensor(),
    
    # Standardizes pixel values (Crucial if using pre-trained models like ResNet)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation Transform have NO AUGMENTATION
test_transforms = transforms.Compose([
    transforms.Resize(256),        # Resize slightly larger first
    transforms.CenterCrop(224),    # Crop the exact center
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------ Load Downloaded Vermont Dataset with Augmentations ------------
base_dataset = ImageFolder(root=dataset_dir)

num_classes = len(base_dataset.classes)
print(f"Total number of classes: {num_classes}")


# Split the dataset (e.g., 80% train, 20% validation)
train_size = int(0.8 * len(base_dataset))
test_size = len(base_dataset) - train_size
train_subset, test_subset = random_split(base_dataset, [train_size, test_size])

# We need a custom wrapper to apply different transforms to each subset
class DatasetWrapper:
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.subset)

# Wrap the subsets with their respective transforms
train_dataset = DatasetWrapper(train_subset, transform=train_transforms)
test_dataset = DatasetWrapper(test_subset, transform=test_transforms)

# Create the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Print dataset sizes to verify loading
print(f"Dataset initialization complete. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
print("------ End Loading Data ------")

# ------------ Train Model ------------
def train_model(model, train_loader, test_loader, epochs=5, lr=0.01, name="Model"):
    """
    Generic training loop with validation. Returns all metrics for comparison.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)

    print(f"\nTraining {name} for {epochs} epochs...")
    start_time = time.time()

    # Metrics to track
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # _, predicted = torch.max(outputs.data, 1)
            # correct += (predicted == labels).sum().item()

            # top k outputs instead of 1
            _, predicted = torch.topk(outputs.data, 5, dim=1)
            # correct if predicted is in top k outputs
            correct += (predicted == labels.view(-1, 1)).sum().item()

            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total


        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # _, predicted = torch.max(outputs.data, 1)
                # correct += (predicted == labels).sum().item()

                # top k outputs instead of 1
                _, predicted = torch.topk(outputs.data, 5, dim=1)
                # correct if predicted is in top k outputs
                correct += (predicted == labels.view(-1, 1)).sum().item()

                total += labels.size(0)

        epoch_val_loss = val_loss / len(test_loader)
        epoch_val_acc = 100 * correct / total

        print(
            f"  Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

    duration = time.time() - start_time
    print(f"{name} - Final Accuracy: {history['val_acc'][-1]:.2f}%, Time: {duration:.2f}s")

    duration = time.time() - start_time
    print(f"{name} — Final Val Acc: {history['val_acc'][-1]:.2f}%, Time: {duration:.2f}s")
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

    # --- Accuracy ---
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "s-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{name} — Accuracy per Epoch")
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
resnet50_exercise = ResNet50_Model(num_classes=num_classes)

# Count parameters and compare with ResNet-18
params_res50 = sum(p.numel() for p in resnet50_exercise.parameters())
print(f"ResNet-50 Parameters: {params_res50:,}")

# Test: ResNet-50 (Scratch Training)
history, duration = train_model(resnet50_exercise, train_loader, test_loader, epochs=10, lr=0.01, name="ResNet-50")

print("------ End Training Model ------")

plot_training_curves(history, name="ResNet50 - Scratch Trained")

"""
Outstanding:
1) Done ---- Figure out how to isolate data to just vermont (Giles)
2) Make sure this data is the right input format for the model (what resoluion are the imaes? need downscaling?)
3) Add code to train/evaluate/setup model
4) Assess performance of normal model (not finetuned) on test set
5) STRATIFY test/train splits
6) Fix issues with nested subsets. This is so messy. Is there a way to just unnest them all every time?

Look into fine tuning (ryan) Do we need to do that?

5) Fine-Tune model on training set & re-assess performance

Misc:
1) Some stats of data (how many species, how many data points, etc). maybe PCA/other visualizations
2) 
"""