# A Place to run filtering on the dataset

import shutil
from pathlib import Path

import torchvision
from shapely.geometry import Point
from torch.utils.data import Subset
from tqdm import tqdm
from torchvision.datasets import inaturalist
import pandas as pd
import geopandas as gpd


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


# Returns True if a given lat/long is inside Vermont
# False Otherwise
def is_in_vermont(lat: float,
                  lon: float,
                  filepath : str = 'data/state_boundary_files/cb_2024_us_all_500k/cb_2024_us_state_500k.shp') -> bool:
    # Get the absolute filepath from the current directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / filepath
    VT_USPS_CODE = 'VT'

    # Read in US Government Shapefile Containing States bounded by polygons
    # This shapefile (when used with geopandas library) lets you send in a point
    # and see which boundary (which state) contains that point
    states = gpd.read_file(DATA_DIR)
    point = Point(lon, lat) # Convert lat/long to a point object expected by the geopandas library

    match = states[states.geometry.contains(point)]
    match_usps_code = match['STUSPS'].iloc[0]

    return VT_USPS_CODE == match_usps_code


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
    print(is_in_vermont( 40.7128, -74.0060))