from config import constants as c
from config.device_config import device, device_name
import utils.data_load_and_config_util as data_config
from models import resnet_50
from utils import evaluate_utils, train_utils

# ------------ Dataset Setup ------------
CURRENT_DATASET_NAME, DATA_PATH = data_config.get_dataset_name_and_path(c.MINI_DATASET)

# ------------ Initialize Loaders ------------
train_loader, test_loader, num_plant_classes = data_config.load_vermont_plant_data(dataset_name=CURRENT_DATASET_NAME,
                                                                                   data_path=DATA_PATH,
                                                                                   device_name=device_name,
                                                                                   batch_size=128)

# ------------ Initialize Model ------------
resnet50_model = resnet_50.ResNet50_Model(num_classes=num_plant_classes)

# ------------ Train Model ------------
history, duration = train_utils.scratch_train_model(resnet50_model,
                                                    train_loader,
                                                    test_loader,
                                                    device,
                                                    device_name,
                                                    epochs=10,
                                                    lr=0.01,
                                                    name="ResNet-50")

# ------------ Evaluate Model ------------
evaluate_utils.plot_training_curves(history, name="ResNet50 - Scratch Trained")
