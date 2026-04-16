"""
Take a model that has already been trained and asses its performance
"""
from config.device_config import device, device_name
import utils.data_load_and_config_util as data_config
from config import constants as c

# ------------ Initial Configuration ------------
MODEL_NAME = "ResNet50 Transfer"
DATASET_USED = c.FULL_DATASET

# ------------ Dataset Setup ------------
CURRENT_DATASET_NAME, DATA_PATH = data_config.get_dataset_name_and_path(DATASET_USED)

# ------------ Initialize Loaders ------------
train_loader, val_loader, num_plant_classes = data_config.load_vermont_plant_data(dataset_name=CURRENT_DATASET_NAME,
                                                                                  data_path=DATA_PATH,
                                                                                  device_name=device_name,
                                                                                  batch_size=128)