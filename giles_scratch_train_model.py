import constants as c
from config.device_config import device, device_name
import config.data_config as data_config
from train_and_evaluate import evaluate_utils, train_utils
from models import resnet_50

# ------------ Dataset Setup ------------
CURRENT_DATASET_NAME, DATA_PATH = data_config.get_dataset_name_path(c.MINI_DATASET)

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
