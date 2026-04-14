import torchvision
from torch.utils.data import random_split, DataLoader
from config import constants as c
from pathlib import Path
from typing import Tuple
import torchvision.transforms as transforms

from utils import dataset_utils


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
    :return: (Train Transform, Val Transform)
    """
    # Training: with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation: no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def load_vermont_plant_data(dataset_name,
                            data_path,
                            device_name,
                            batch_size=128) -> Tuple[DataLoader, DataLoader, int]:

    print("------ BEGIN Loading Data ------")
    VAL_DATASET_NAME = c.VAL_DATASET
    VAL_DATA_PATH = str(Path(__file__).resolve().parent.parent.parent / "data")

    # Define the data transformations
    train_transform, val_transform = get_test_transfer_transforms()

    # Delete any lingering MacOS Preview Files (these break the torchvision loaders)
    dataset_utils.delete_ds_store(data_path)

    # Load the full training dataset
    full_dataset = torchvision.datasets.INaturalist(root=data_path,
                                                    version=dataset_name,
                                                    target_type="full",
                                                    transform=train_transform,
                                                    download=False)

    # Load the validation training dataset
    val_dataset = torchvision.datasets.INaturalist(root=VAL_DATA_PATH,
                                                   version=VAL_DATASET_NAME,
                                                   target_type="full",
                                                   transform=val_transform,
                                                   download=False)

    # Subset the dataset further to only include Plants found in Vermont
    vermont_plant_dataset, vermont_cat_ids = dataset_utils.return_species_relevant_to_vermont(dataset=full_dataset,
                                                                                              dataset_name=dataset_name,
                                                                                              kingom_name="Plantae")

    # Subset the validation dataset down to just species in vermont
    vermont_val_plant_dataset = dataset_utils.filter_by_cat_ids(val_dataset,
                                                                cat_ids=vermont_cat_ids,
                                                                kingom_name="Plantae")

    # Flatten nested subsets and create contiguous integer labels
    train_set = dataset_utils.FlatDataset(vermont_plant_dataset)
    val_set = dataset_utils.FlatDataset(vermont_val_plant_dataset, cat_id_to_label=train_set.cat_id_to_label)

    num_plant_classes = train_set.num_classes

    # Create DataLoaders for efficient batch processing using the whole dataset
    use_cuda = (device_name == 'cuda')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4 if use_cuda else 0,
                              pin_memory=use_cuda)

    # Test loader uses the test set for final evaluation
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=4 if use_cuda else 0,
                             pin_memory=use_cuda)

    # Print dataset sizes to verify loading
    print(f"Dataset initialization complete. Train: {len(train_set)}, Val: {len(val_set)}")
    print("------ END Loading Data ------")

    return train_loader, val_loader, num_plant_classes
