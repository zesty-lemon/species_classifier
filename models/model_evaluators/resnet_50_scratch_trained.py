from config import constants as c
from config.device_config import device, device_name
import utils.data_load_and_config_util as data_config
from model_definitions import resnet_50_scratch_trained
from utils import evaluate_utils
from models.model_utils import model_utils, train_utils

# ------------ Initial Configuration ------------
MODEL_NAME = "ResNet50 Scratch Trained Augmentation Early Stopping"
DATASET_USED = c.FULL_DATASET

# ------------ Dataset Setup ------------
CURRENT_DATASET_NAME, DATA_PATH = data_config.get_dataset_name_and_path(DATASET_USED)

# ------------ Initialize Loaders ------------
train_loader, val_loader, num_plant_classes = data_config.load_vermont_plant_data(dataset_name=CURRENT_DATASET_NAME,
                                                                                  data_path=DATA_PATH,
                                                                                  device_name=device_name,
                                                                                  batch_size=128)

# ------------ Initialize Model ------------
resnet50_model = resnet_50_scratch_trained.ResNet50_Model(num_classes=num_plant_classes)

# ------------ Train Model ------------
history, duration = train_utils.train_model(resnet50_model,
                                            train_loader,
                                            val_loader,
                                            device,
                                            device_name,
                                            epochs=30,
                                            lr=0.05,
                                            name=MODEL_NAME)

# ------------ Evaluate Model ------------
evaluate_utils.plot_training_curves(history,
                                    dataset_name=CURRENT_DATASET_NAME,
                                    name=MODEL_NAME)

evaluate_utils.generate_performance_report(model=resnet50_model,
                                           val_loader=val_loader,
                                           device=device,
                                           device_name=device_name,
                                           history=history,
                                           dataset_name=CURRENT_DATASET_NAME,
                                           name=MODEL_NAME,
                                           annotation="Trained on Full Dataset")

# ------------ Persist Trained Model ------------
model_utils.persist_trained_model(model=resnet50_model,
                                  dataset_name=CURRENT_DATASET_NAME,
                                  name=MODEL_NAME)