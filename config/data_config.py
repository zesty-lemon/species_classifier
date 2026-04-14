import torchvision
from torch.utils.data import random_split, DataLoader
from config import constants as c
from pathlib import Path
from typing import Tuple
import torchvision.transforms as transforms
import scripts.file_operations
import scripts.dataset_utils

def get_dataset_name_and_path(dataset_name: str) -> Tuple[str, str]:
    """
    Dynamically get the path to the desired dataset. Check local, then external volumes
    :param dataset_name:
    :return:
    """
    local_directory_path = Path(c.MINI_LOCAL_DATA_DIR, dataset_name)
    external_directory_path = Path(c.EXTERNAL_DATA_DIR, dataset_name)
    system_directory_path = Path(c.SYSTEM_DATA_DIR, dataset_name)

    DATA_PATH = ""
    if local_directory_path.is_dir():
        DATA_PATH = c.MINI_LOCAL_DATA_DIR
        print(f"Loading Dataset {dataset_name} from path {local_directory_path}")
    elif system_directory_path.is_dir():
        DATA_PATH = c.SYSTEM_DATA_DIR
        print(f"Loading Dataset {dataset_name} from path {system_directory_path}")
    elif external_directory_path.is_dir():
        DATA_PATH = c.EXTERNAL_DATA_DIR
        print(f"Loading Dataset {dataset_name} from path {external_directory_path}")
    else:
        print(f"ERROR - data for dataset {dataset_name} not found in directory "
              f"{local_directory_path} OR {system_directory_path} OR {external_directory_path} "
              f"\n Check Paths & Dataset Name")

    return dataset_name, DATA_PATH


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


def load_vermont_plant_data(dataset_name,
                            data_path,
                            device_name,
                            batch_size=128) -> Tuple[DataLoader, DataLoader, int]:

    print("------ BEGIN Loading Data ------")
    # Define the data transformations
    test_transform, transfer_transform = get_test_transfer_transforms()

    # Delete any lingering MacOS Preview Files (these break the torchvision loaders)
    scripts.file_operations.delete_ds_store(data_path)

    # Download and load the full training dataset
    full_dataset = torchvision.datasets.INaturalist(root=data_path,
                                                    version=dataset_name,
                                                    target_type="full",
                                                    transform=transfer_transform,
                                                    download=False)

    # Subset the dataset further to only include Plants found in Vermont
    vermont_plant_dataset = scripts.dataset_utils.return_species_relevant_to_vermont(dataset=full_dataset,
                                                                                     kingom_name="Plantae")

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
    print("------ END Loading Data ------")

    return train_loader, test_loader, num_plant_classes