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
scripts.file_operations.delete_ds_store(DATA_PATH)

# Download and load the full training dataset
full_dataset = torchvision.datasets.INaturalist(root=DATA_PATH,
                                             version='2021_train_mini',
                                             target_type="full",
                                             transform = transfer_transform,
                                             download = False)

# Subset the dataset to only include plants
plant_dataset = scripts.dataset_utils.return_specified_kingdom(full_dataset=full_dataset, kingom_name="Plantae")

# # Subset the dataset further to only include Vermont images
plant_dataset = scripts.dataset_utils.return_vermont_images(plant_dataset)

train_size = int(0.8 * len(plant_dataset))
test_size = len(plant_dataset) - train_size
train_set, test_set = random_split(plant_dataset, [train_size, test_size])

# Create DataLoaders for efficient batch processing using the whole dataset
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
# Test loader uses the test set for final evaluation
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
# Print dataset sizes to verify loading
print(f"Dataset initialization complete. Train: {len(train_set)}, Test: {len(test_set)}")
print("------ END Loading Data ------")

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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

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

    return history['train_acc'][-1], history['train_loss'][-1], history['val_acc'][-1], history['val_loss'][
        -1], duration


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
print("------ BEGIN Training Model ------")
resnet50_exercise = ResNet50_Model(num_classes=10)

# Count parameters and compare with ResNet-18
params_res50 = sum(p.numel() for p in resnet50_exercise.parameters())
print(f"ResNet-50 Parameters: {params_res50:,}")

# Test: ResNet-50 (Scratch Training)
train_acc, train_loss, val_acc, val_loss, t = train_model(resnet50_exercise, train_loader, test_loader, epochs=2, lr=0.01, name="ResNet-50")

print("------ END Training Model ------")

results['ResNet-50'] = {
    'train_acc': train_acc,
    'train_loss': train_loss,
    'val_acc': val_acc,
    'val_loss': val_loss,
    'params': params_res50,
    'time': t,
    'size_mb': params_res50 * 4 / (1024**2)
}

"""
Outstanding:
1) Done ---- Figure out how to isolate data to just vermont (Giles)
2) Make sure this data is the right input format for the model (what resoluion are the imaes? need downscaling?)
3) Add code to train/evaluate/setup model
4) Assess performance of normal model (not finetuned) on test set
5) 

Look into fine tuning (ryan) Do we need to do that?

5) Fine-Tune model on training set & re-assess performance

Misc:
1) Some stats of data (how many species, how many data points, etc). maybe PCA/other visualizations
2) 
"""