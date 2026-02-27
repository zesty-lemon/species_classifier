# A Place to run filtering on the dataset
import json
import shutil
from pathlib import Path

import torchvision
from shapely.geometry import Point
from torch.utils.data import Subset
from tqdm import tqdm
from torchvision.datasets import inaturalist
import pandas as pd
import geopandas as gpd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

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


def get_vermont_indices(dataset,
                        shapefile_path : str = 'data/state_boundary_files/cb_2024_us_all_500k/cb_2024_us_state_500k.shp'):
    """Batch-check all images and return indices inside Vermont."""

    # Get the absolute filepath from the current directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / shapefile_path

    states = gpd.read_file(DATA_DIR)
    vt_geom = states[states['STUSPS'] == 'VT'].geometry.iloc[0]

    indices = []

    image_annotations = read_image_annotations_from_file()

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


def return_vermont_images(dataset):
    """Filter dataset indices to only images geolocated in Vermont."""
    print("----- BEGIN filtering dataset to only Vermont -----")
    vermont_indices = []

    vermont_indices = get_vermont_indices(dataset=dataset)

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
def read_image_annotations_from_file(annotation_filepath: str = "data/2021_train_mini_annotations/train_mini.json") -> dict[
    str, tuple[float, float]]:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / annotation_filepath

    with open(DATA_DIR) as f:
        data = json.load(f)

    # Grab just the image name, the coordinates, and returns them
    coords = {Path(img['file_name']).name: (img['latitude'], img['longitude']) for img in data['images']}

    return coords

if __name__ == "__main__":
    read_image_annotations_from_file()
    print(is_in_vermont( 40.7128, -74.0060))

    """Giles Todo:
    fix rest of type hints
    add toggle for what to do with images with no gps data
    factor in location uncertainty?
    Abstract data preprocessing out into its own script that runs before main script
    use get_vermont_indices since it doesn't have to reload shapefile every time
    Put constants all in one place
    """