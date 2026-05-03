import io
import os

import joblib
import torch
import torch.nn as nn

from config import constants as c


def persist_trained_model(model: nn.Module, dataset_name: str, name: str = "Model") -> None:
    """
    Save trained models to disk
    :param model: Trained model object
    :param dataset_name: The name of the dataset (to make directory)
    :param name: Name of the model, i.e "Resnet-50"
    :return: None
    """
    print("----- BEGIN Saving Model to Disk -----")
    name = name.replace(" ", "_")
    directory = c.PROJECT_ROOT / "models" / "trained_models" / name / dataset_name
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model, f'{directory}/{name}_model.joblib')
    print(f"Model: {name} saved to path: {directory}/{name}_model.joblib")
    print("----- END Saving Model to Disk -----")


#Use for models trained on mac
def get_trained_model(path_to_model: str) -> nn.Module:
    """

    :param path_to_model: Absolute filepath
    :return: Trained Model
    """
    trained_model = joblib.load(path_to_model)
    return trained_model


# use for models trained on PC
def get_cuda_trained_model(path_to_model: str) -> nn.Module:
    """
    :param path_to_model: Absolute filepath
    :return: Trained Model
    """
    # joblib.load doesn't expose map_location, so override the nested torch.load
    # that fires inside pickled tensor storages. Lets CUDA-trained checkpoints
    # load on machines without CUDA (e.g. MPS/CPU).
    original_load_from_bytes = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False
    )
    try:
        trained_model = joblib.load(path_to_model)
    finally:
        torch.storage._load_from_bytes = original_load_from_bytes
    return trained_model