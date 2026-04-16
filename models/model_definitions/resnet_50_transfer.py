import torch.nn as nn
import torchvision.models as models

"""
ResNet-50 Transfer Learning Model
Loads pretrained ImageNet weights, freezes the backbone, and replaces the classification head.
"""
def create_transfer_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) # use IMAGENET1K_V2 pretrained weights

    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classification head with a new one for the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
