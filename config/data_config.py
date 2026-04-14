from typing import Tuple

import constants as c
from pathlib import Path

def get_dataset_name_path(dataset_name: str) -> Tuple[str, str]:
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