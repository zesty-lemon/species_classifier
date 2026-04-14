import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import matplotlib.pyplot as plt
import numpy as np

from utils import dataset_utils

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
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

# Delete any lingering MacOS Preview Files (these break the torchvision loaders)
dataset_utils.delete_ds_store(DATA_PATH)

# Download and load the full training dataset
full_dataset = torchvision.datasets.INaturalist(root=DATA_PATH,
                                             version='2021_train_mini',
                                             target_type="full",
                                             transform = transfer_transform,
                                             download = False)

# Subset the dataset to only include plants
plant_dataset = dataset_utils.return_specified_kingdom(full_dataset=full_dataset, kingom_name="Plantae")

# # # Subset the dataset further to only include Vermont images
# plant_dataset = scripts.dataset_utils.return_vermont_images(plant_dataset)

# Flatten nested subsets and create contiguous integer labels
flat_dataset = dataset_utils.FlatDataset(plant_dataset)
num_plant_classes = flat_dataset.num_classes

print(f"Num Classes: {num_plant_classes}")

# train_size = int(0.8 * len(flat_dataset))
# test_size = len(flat_dataset) - train_size
# transfer_train_set, transfer_test_set = random_split(flat_dataset, [train_size, test_size])

train_size = int(0.001 * len(flat_dataset))
test_size = int(0.001 * len(flat_dataset))
junk_size = len(flat_dataset) - train_size - test_size
transfer_train_set, transfer_test_set, junk_set = random_split(flat_dataset, [train_size, test_size, junk_size])

# Create DataLoaders for efficient batch processing using the whole dataset
transfer_train_loader = DataLoader(transfer_train_set, batch_size=64, shuffle=True) # Smaller batch size due to larger images
# Test loader uses the test set for final evaluation
transfer_test_loader = DataLoader(transfer_test_set, batch_size=64, shuffle=False)# Print dataset sizes to verify loading
print(f"Dataset initialization complete. Train: {len(transfer_train_set)}, Test: {len(transfer_test_set)}")
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
    plt.show()

def get_transfer_model(model_name='resnet18', num_classes=10, feature_extract=True, weights_name='DEFAULT'):
    """
    A generalized function to perform transfer learning on any torchvision model.
    1. Loads pre-trained weights.
    2. Freezes the backbone (optional).
    3. Replaces the final classifier for the target number of classes.

    NOTE: Since our data pipeline converts grayscale to 3-channel RGB using
    transforms.Grayscale(num_output_channels=3), we do NOT modify the input layer.
    The pre-trained weights work perfectly with 3-channel input.
    """
    # 1. Dynamically load the model from torchvision.models
    if hasattr(models, model_name):
        model_func = getattr(models, model_name)
        # IMPORTANT: For GoogLeNet, we must set transform_input=False
        # because it expects 3-channel ImageNet normalization by default.
        if model_name == 'googlenet':
            model = model_func(weights=weights_name, transform_input=False)
        else:
            model = model_func(weights=weights_name)
    else:
        raise ValueError(f"Model {model_name} not found in torchvision.models")

    # 2. Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # 3. Keep the first convolutional layer unchanged
    # Since our data pipeline converts grayscale images to 3-channel RGB,
    # the pre-trained first layer (which expects 3 channels) works perfectly.
    # This preserves all the pre-trained knowledge from ImageNet without any adaptation.

    # 4. Replace the final classifier (the 'head')
    if hasattr(model, 'fc'):  # ResNet, GoogLeNet
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
    elif hasattr(model, 'classifier'):  # VGG
        if isinstance(model.classifier, nn.Sequential):
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            model.classifier[-1].weight.requires_grad = True
            model.classifier[-1].bias.requires_grad = True
        else:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            model.classifier.weight.requires_grad = True
            model.classifier.bias.requires_grad = True

    # 5. If not feature extracting, unfreeze the whole model (Fine-tuning)
    if not feature_extract:
        for param in model.parameters():
            param.requires_grad = True

    return model

# Initialize
print("------ BEGIN Training Model ------")
# Evaluation 4: Transfer Learning with Fine-Tuning (ResNet-18)
# For better accuracy, we use feature_extract=False to perform fine-tuning
transfer_resnet = get_transfer_model('resnet50', num_classes=num_plant_classes, feature_extract=True)

# Count trainable parameters
trainable_params = sum(p.numel() for p in transfer_resnet.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in transfer_resnet.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

# Train the model with a smaller learning rate for fine-tuning
# We also use a slightly larger number of epochs if needed, but 5 is a good start.
history, duration = train_model(
    transfer_resnet,
    transfer_train_loader,
    transfer_test_loader,
    epochs=5,
    lr=0.001, # Smaller LR for fine-tuning to preserve pre-trained features
    name="Transfer-ResNet50"
)

plot_training_curves(history, name="ResNet50 - Fine Tuned")