from pathlib import Path

# Absolute project root — works regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Two places to look for large datasets (training images, validation images, annotations)
# Priority 1: /data directory one level above the project
LOCAL_DATA_DIR = PROJECT_ROOT.parent / "data"
# Priority 2: /data directory on removable volume
EXTERNAL_DATA_DIR = Path("/Volumes/giDrive/data")

# Project-internal data (shapefiles, mini dataset annotations — always in the repo)
INTERNAL_DATA_DIR = PROJECT_ROOT / "data"

# Dataset version names
MINI_DATASET = "2021_train_mini"
FULL_DATASET = "2021_train"
VAL_DATASET = "2021_valid"


def resolve_data_dir(dataset_name: str) -> Path:
    """Find the data directory containing the given dataset.
    Checks local first, then the external volume."""
    for base in (LOCAL_DATA_DIR, EXTERNAL_DATA_DIR):
        candidate = base / dataset_name
        if candidate.is_dir():
            print(f"Found dataset '{dataset_name}' at {candidate}")
            return base
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found in:\n"
        f"  {LOCAL_DATA_DIR}\n"
        f"  {EXTERNAL_DATA_DIR}\n"
        f"Check that the data exists in one of these locations."
    )
