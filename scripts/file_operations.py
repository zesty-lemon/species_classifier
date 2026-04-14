import os
import sys
from pathlib import Path

# Macos makes junk files called .DS_Store files whenever you open an image in Finder
# These files break the way torchvision opens the iNaturalist data
# before opening the inaturalist data, we have to clean them up
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


if __name__ == "__main__":
    delete_ds_store('data/')