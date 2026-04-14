# A Place to run filtering on the dataset
import json
import os
import shutil
import sys
from pathlib import Path

import torchvision
from shapely.geometry import Point
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from torchvision.datasets import inaturalist
import pandas as pd
import geopandas as gpd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def delete_ds_store(target_dir):
    """
    Delete any lingering MacOS Preview Files (these break the torchvision loaders)
    :param target_dir: Root directory of data
    :return: None
    """
    # get the project root (fixes weird issues with relative filepaths by making them absolute)
    print("Deleting .DS_Store files (MacOS Preview Files)")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / target_dir

    if not os.path.isdir(DATA_DIR):
        print(f"Error: '{DATA_DIR}' is not a valid directory.")
        sys.exit(1)

    deleted = 0
    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            if filename == ".DS_Store":
                filepath = os.path.join(root, filename)
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filepath}")
                    deleted += 1
                except OSError as e:
                    print(f"Error deleting {filepath}: {e}")

    if deleted == 0:
        print("No .DS_Store files found.")
    else:
        print(f"\nDone. Deleted {deleted} .DS_Store file(s).")


# Subsets of Torchvision datasets are disgusting
# they contain an index and a link to the original full dataset
# but subsetting a subset makes a link to the subset which links to the original set
# this becomes DISGUSTING almost immediately.
# this method is designed to fix that
class FlatDataset(Dataset):
    """Unwraps nested Subsets into a flat dataset with contiguous integer labels."""

    def __init__(self, subset):
        # Resolve all Subset layers down to the base INaturalist dataset
        indices = list(range(len(subset)))
        ds = subset
        while isinstance(ds, Subset):
            indices = [ds.indices[i] for i in indices]
            ds = ds.dataset

        self.base_dataset = ds
        self.indices = indices

        # Build a mapping from sparse cat_ids to contiguous [0, num_classes-1]
        cat_ids = set()
        for idx in self.indices:
            cat_id, _ = ds.index[idx]
            cat_ids.add(cat_id)

        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(sorted(cat_ids))}
        self.label_to_category = {i: ds.all_categories[cat_id] for cat_id, i in self.cat_id_to_label.items()}
        self.num_classes = len(self.cat_id_to_label)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        cat_id, _ = self.base_dataset.index[real_idx]
        image, _ = self.base_dataset[real_idx]
        label = self.cat_id_to_label[cat_id]
        return image, label


def convert_dms_to_decimal(dms_tuple: tuple, ref: str) -> float | None:
    """Convert (degrees, minutes, seconds) + reference to decimal degrees."""
    try:
        degrees, minutes, seconds = [float(v) for v in dms_tuple]
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal
    except (ValueError, TypeError):
        return None


# Subset a full inaturalist dataset by a desired kingdom
def return_specified_kingdom(full_dataset: torchvision.datasets.INaturalist,
                             kingom_name: str = "plantae") -> Subset[torchvision.datasets.INaturalist]:

    # Find category IDs where kingdom is "Plantae"
    plantae_cat_ids = set()
    for index in range(len(full_dataset.all_categories)):
        category = full_dataset.all_categories[index]

        if kingom_name in category:
            plantae_cat_ids.add(index)

    # Find dataset indices (list) that belong to those categories
    plantae_indices = [
        i for i, (cat_id, _) in enumerate(full_dataset.index)
        if cat_id in plantae_cat_ids
    ]

    # Create the filtered subset
    plantae_dataset = Subset(full_dataset, plantae_indices)

    return plantae_dataset


def check_any_in_vermont(base_dataset: torchvision.datasets.INaturalist,
                         cat_id: int,
                         real_indices: list[int],
                         annotations: dict[str, tuple[float, float]],
                         vt_geom) -> bool:
    """Check if any image for a given cat_id is geolocated in Vermont."""
    # Filter to just the real indices that belong to this cat_id
    indices_for_cat = [idx for idx in real_indices if base_dataset.index[idx][0] == cat_id]

    # Go category by category, grab the annotations for the filename (the individual images) check if in vermont
    for idx in indices_for_cat:
        _, filename = base_dataset.index[idx]
        if filename in annotations:
            lat, lon = annotations[filename]
            if lat is not None and lon is not None:
                if vt_geom.contains(Point(lon, lat)):
                    return True

    return False


def return_species_relevant_to_vermont(dataset: torchvision.datasets.INaturalist,
                                       dataset_name: str = "2021_train",
                                       kingom_name: str = "plantae") -> Subset:
    """Return all images of species that have at least one observation in Vermont."""

    plant_dataset = return_specified_kingdom(full_dataset=dataset, kingom_name=kingom_name)

    # Unwrap the Subset to get the base dataset and real indices
    ds = plant_dataset
    indices = list(range(len(ds)))
    while isinstance(ds, Subset):
        indices = [ds.indices[i] for i in indices]
        ds = ds.dataset

    # Load annotations and Vermont geometry once
    annotations = read_image_annotations_from_file(dataset_name=dataset_name)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    shapefile_path = PROJECT_ROOT / 'data/state_boundary_files/cb_2024_us_all_500k/cb_2024_us_state_500k.shp'
    states = gpd.read_file(shapefile_path)
    vt_geom = states[states['STUSPS'] == 'VT'].geometry.iloc[0]

    # Get unique plant cat_ids present in the subset
    plant_cat_ids = set(ds.index[idx][0] for idx in indices)

    # Check each species for Vermont presence
    vermont_cat_ids = set()
    for cat_id in tqdm(plant_cat_ids, desc="Checking species for Vermont presence", unit="species"):
        if check_any_in_vermont(ds, cat_id, indices, annotations, vt_geom):
            vermont_cat_ids.add(cat_id)

    print(f"Species found in Vermont: {len(vermont_cat_ids)} / {len(plant_cat_ids)}")

    # Return ALL images of Vermont-relevant species (not just the Vermont ones)
    vermont_species_indices = [
        i for i, idx in enumerate(indices)
        if ds.index[idx][0] in vermont_cat_ids
    ]

    return Subset(plant_dataset, vermont_species_indices)

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
    # The inversion (lon, lat) instead of (lat, long) is correct, Point expects X, Y
    point = Point(lon, lat) # Convert lat/long to a point object expected by the geopandas library

    match = states[states.geometry.contains(point)]
    if len(match['STUSPS']) == 0:
        # Point is OUT OF BOUNDS (not in ANY state)
        return False
    else:
        match_usps_code = match['STUSPS'].iloc[0]
        return VT_USPS_CODE == match_usps_code


def get_lat_lon_from_annotations(dataset, idx, annotations: dict[str, tuple[float, float]]) -> tuple[float | None, float | None]:
    # the dataset is itself a subset
    # this subset object is just a wrapper, containing the original dataset and a list of indicies
    # to get to the actual dataset, we have to go down a layer

    # Unwrap Subset layers to get to the INaturalist dataset and the real index
    real_idx = idx
    ds = dataset
    while isinstance(ds, Subset):
        real_idx = ds.indices[real_idx]
        ds = ds.dataset

    # ds is now the INaturalist object, real_idx is the index into it
    cat_id, filename = ds.index[real_idx]

    lat, lon = annotations[filename]
    return lat, lon


def get_vermont_indices(dataset, dataset_name: str = "2021_train_mini",
                        shapefile_path : str = 'data/state_boundary_files/cb_2024_us_all_500k/cb_2024_us_state_500k.shp'):
    """Batch-check all images and return indices inside Vermont."""

    # Get the absolute filepath from the current directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / shapefile_path

    states = gpd.read_file(DATA_DIR)
    vt_geom = states[states['STUSPS'] == 'VT'].geometry.iloc[0]

    indices = []

    image_annotations = read_image_annotations_from_file(dataset_name=dataset_name)

    images_missing_lat_long = 0
    images_in_vermont = 0
    images_outside_vermont = 0

    for idx in tqdm(range(len(dataset)), unit="images", desc="Filtering Image Locations"):
        lat, lon = get_lat_lon_from_annotations(dataset, idx, image_annotations)
        if lat is not None and lon is not None:
            if vt_geom.contains(Point(lon, lat)):
                indices.append(idx)
                images_in_vermont +=1
            else: images_outside_vermont += 1
        else:
            images_missing_lat_long+=1

    print(f"Total Images:             {len(dataset)}")
    print(f"Images in Vermont:        {images_in_vermont} ({(images_in_vermont/len(dataset) * 100):.2f}%)")
    print(f"Images Outside Vermont:   {images_outside_vermont} ({(images_outside_vermont/len(dataset) * 100):.2f}%)")
    print(f"Images Missing Locations: {images_missing_lat_long} ({(images_missing_lat_long/len(dataset) * 100):.2f}%)")

    return indices


def return_vermont_images(dataset, dataset_name: str = "2021_train_mini"):
    """Filter dataset indices to only images geolocated in Vermont."""
    print("----- BEGIN filtering dataset to only Vermont -----")

    vermont_indices = get_vermont_indices(dataset=dataset, dataset_name=dataset_name)

    print("----- END filtering dataset to only Vermont -----")
    return Subset(dataset, vermont_indices)

"""
Json Structure for Annotations:
  {
    "id": 0,
    "width": 500,
    "height": 500,
    "file_name": "train_mini/02912_Animalia_Chordata_.../d615f184-....jpg",
    "license": 0,
    "rights_holder": "Ken-ichi Ueda",
    "date": "2010-07-14 20:19:00+00:00",
    "latitude": 43.83486,
    "longitude": -71.22231,
    "location_uncertainty": 77
  }
  """
def read_image_annotations_from_file(annotation_filepath: str = None, dataset_name: str = "2021_train_mini") -> dict[
    str, tuple[float, float]]:
    if annotation_filepath is None:
        # Map dataset name to its annotation file
        annotation_files = {
            "2021_train_mini": "data/2021_train_mini_annotations/train_mini.json",
            "2021_train": "../data/2021_train_annotations/train.json",
        }
        annotation_filepath = annotation_files[dataset_name]
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / annotation_filepath

    with open(DATA_DIR) as f:
        data = json.load(f)

    # Grab just the image name, the coordinates, and returns them
    coords = {Path(img['file_name']).name: (img['latitude'], img['longitude']) for img in data['images']}

    return coords


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
    read_image_annotations_from_file()
    print(is_in_vermont( 40.7128, -74.0060))

    """
    Subset by Relavent To Vermont
    1) Subset to Plants first
    2) Within plants, go species by species
    3) If an example is inside of Vermont, add ALL instances of species to index, not just that single one. 
    """