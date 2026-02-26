# A Place to run filtering on the dataset

import shutil
from pathlib import Path

import torchvision
from sympy import Point
from torch.utils.data import Subset
from tqdm import tqdm
from torchvision.datasets import inaturalist
# import geopandas as gpd


# Subset a full inaturalist dataset by a desired kingdom
def return_specified_kingdom(full_dataset: torchvision.datasets.INaturalist,
                             kingom_name: str = "plantae") -> Subset[torchvision.datasets.INaturalist]:

    # Find category IDs where kingdom is "Plantae"
    plantae_cat_ids = set()
    for index in range(len(full_dataset.all_categories)):
        category = full_dataset.all_categories[index]

        if'Plantae' in category:
            plantae_cat_ids.add(index)

    # Find dataset indices (list) that belong to those categories
    plantae_indices = [
        i for i, (cat_id, _) in enumerate(full_dataset.index)
        if cat_id in plantae_cat_ids
    ]

    # Create the filtered subset
    plantae_dataset = Subset(full_dataset, plantae_indices)

    return plantae_dataset


# # From https://austinhenley.com/blog/coord2state.html#:~:text=How%20does%20it%20work?,the%20borders%20are%20really%20detailed.
#
# def is_in_vermont():
#     states = gpd.read_file("tl_2024_us_state.shp")
#     lon, lat = -74.0060, 40.7128  # New York
#     match = states[states.geometry.contains(Point(lon, lat))].iloc[0]["NAME"]


# Filter Dataset by Kingdom (Delete Directories Not In Desired Kingdom)
def remove_unwanted_kingdoms(kingdom_to_keep: str, data_filepath: str = '/data/2021_train_mini'):
    print(f"----- BEGIN Filtering By Kingdom = {kingdom_to_keep} -----")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / data_filepath

    directory_path = Path(DATA_DIR)

    # Collect directories for tqdm knows the total
    subdirs = [p for p in directory_path.iterdir() if p.is_dir()]

    num_directories_kept = 0
    num_directories_deleted = 0

    for path in tqdm(subdirs, desc="Filtering directories", unit="dir"):
        if kingdom_to_keep not in str(path):
            try:
                shutil.rmtree(path)
                num_directories_deleted += 1
            except OSError as e:
                print(f"Error: {path} : {e.strerror}")
        else:
            num_directories_kept += 1

    print("----- Filtering Job Report -----")
    print(f"Directories Kept:    {num_directories_kept}")
    print(f"Directories Deleted: {num_directories_deleted}")
    print(f"----- END Filtering By Kingdom = {kingdom_to_keep} -----")


if __name__ == "__main__":
    remove_unwanted_kingdoms(kingdom_to_keep="Plantae",
                             data_filepath = 'data/2021_train_mini')