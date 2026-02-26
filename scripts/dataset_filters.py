import shutil
from pathlib import Path

# Filter Dataset by Kingdom (Delete Directories Not In Desired Kingdom)
def remove_unwanted_kingdoms(kingdom_to_keep: str, data_filepath: str = '/data/2021_train_mini'):
    print(f"----- BEGIN Filtering By Kingdom = {kingdom_to_keep} -----")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / data_filepath

    directory_path = Path(DATA_DIR)

    for path in directory_path.iterdir():
        if path.is_dir():
            if kingdom_to_keep not in str(path):
                try:
                    shutil.rmtree(directory_path)
                    print(f"Directory '{directory_path}' and all its contents have been removed.")
                except OSError as e:
                    # Print an error if it occurs (permission errors)
                    print(f"Error: {directory_path} : {e.strerror}")

    print(f"----- END Filtering By Kingdom = {kingdom_to_keep} -----")


if __name__ == "__main__":
    remove_unwanted_kingdoms(kingdom_to_keep="Plantae",
                             data_filepath = 'data/2021_train_mini')