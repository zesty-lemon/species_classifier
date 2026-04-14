from typing import Tuple

import torchvision.transforms as transforms


def get_test_transfer_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get test & Transfer transforms
    :return: (Test Transform, Transfer Transform)
    """
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Transform images to resnet standard
    transfer_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to ImageNet standard
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    return test_transform, transfer_transform