import shutil
from pathlib import Path
from tqdm import tqdm

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