"""
Take a model that has already been trained and asses its performance
"""
import numpy as np
import torch
from PIL import Image
from numpy import average, std, median, percentile
from tqdm import tqdm

from config.device_config import device, device_name
import utils.data_load_and_config_util as data_config
from config import constants as c
from model_utils.model_utils import get_trained_model, persist_trained_model

# ------------ Initial Configuration ------------
MODEL_NAME = "ResNet50 Scratch Trained"
MODEL_PATH = "/Users/giles/Documents/Grad_School/Spring_2026/deep_learning/Project/species_classifier/models/trained_models/ResNet50_Scratch_Trained/2021_train/ResNet50_Scratch_Trained_model.joblib"
DATASET_USED = c.FULL_DATASET # Need to leave this even though it isn't being used, need it for category indexes
TOP_K = 5

# ------------ Dataset Setup ------------
CURRENT_DATASET_NAME, DATA_PATH = data_config.get_dataset_name_and_path(DATASET_USED)

# ------------ Initialize Loaders ------------
_, val_loader, num_plant_classes = data_config.load_vermont_plant_data(dataset_name=CURRENT_DATASET_NAME,
                                                                                  data_path=DATA_PATH,
                                                                                  device_name=device_name,
                                                                                  batch_size=128)

trained_model = get_trained_model(MODEL_PATH)
trained_model = trained_model.to(device)
trained_model.eval()

# ------------ Classify a Single Image ------------
IMAGE_PATH = '/Users/giles/Library/CloudStorage/GoogleDrive-gileslemmon@gmail.com/My Drive/deep_learning/data/2021_valid/09346_Plantae_Tracheophyta_Magnoliopsida_Rosales_Rhamnaceae_Rhamnus_humboldtiana/1c31a45a-7e99-4568-b760-f127e07313c7.jpg'

# image has to be transformed the same way as original input images
_, val_transform = data_config.get_test_transfer_transforms()

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = val_transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    logits = trained_model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    topk_probs, topk_idx = probs.topk(TOP_K, dim=1)

label_to_category = val_loader.dataset.label_to_category

print(f"\nPredictions for {IMAGE_PATH}:")
for rank, (idx, prob) in enumerate(zip(topk_idx[0].tolist(), topk_probs[0].tolist()), start=1):
    print(f"  {rank}. {label_to_category[idx]} — {prob * 100:.2f}%")

correct_classification_top_confidences = []
incorrect_classification_top_confidences = []
correct_classification_margins = []
incorrect_classification_margins = []

# count of times top classification was correct but image was in top K
count_incorrect_class_but_in_top_k = 0
count_incorrect_top_classification = 0

# store per-image stats
# margin, top 1 confidence, top class correct? was correct class in top k classes?
records = []

for image, label in tqdm(val_loader.dataset, desc="Evaluating", unit=" images"):
    input_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = trained_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        topk_probs, topk_idx = probs.topk(TOP_K, dim=1)

        topk_probs_list = topk_probs.tolist()[0]
        topk_idx_list = topk_idx.tolist()[0]
        margin = topk_probs_list[0] - topk_probs_list[1]

        if topk_idx_list[0] == label: # Classification is CORRECT if label matches top class label
            correct_classification_top_confidences.append(topk_probs_list[0])
            correct_classification_margins.append(margin)
        else:
            incorrect_classification_top_confidences.append(topk_probs_list[0])
            incorrect_classification_margins.append(margin)
            count_incorrect_top_classification = count_incorrect_top_classification + 1
            if label in topk_idx_list:
                count_incorrect_class_but_in_top_k = count_incorrect_class_but_in_top_k + 1

        records.append((
            margin,
            topk_probs_list[0],
            top1_correct := (topk_idx_list[0] == label),
            label in topk_idx_list,
        ))

avg_top_confidence_correct = average(correct_classification_top_confidences)
std_top_confidence_correct = std(correct_classification_top_confidences)
max_top_confidence_correct = max(correct_classification_top_confidences)
min_top_confidence_correct = min(correct_classification_top_confidences)
median_top_confidence_correct = median(correct_classification_top_confidences)
q25_top_confidence_correct = percentile(correct_classification_top_confidences, 25)
q75_top_confidence_correct = percentile(correct_classification_top_confidences, 75)
iqr_top_confidence_correct = q75_top_confidence_correct - q25_top_confidence_correct

avg_top_confidence_incorrect = average(incorrect_classification_top_confidences)
std_top_confidence_incorrect = std(incorrect_classification_top_confidences)
max_top_confidence_incorrect = max(incorrect_classification_top_confidences)
min_top_confidence_incorrect = min(incorrect_classification_top_confidences)
median_top_confidence_incorrect = median(incorrect_classification_top_confidences)
q25_top_confidence_incorrect = percentile(incorrect_classification_top_confidences, 25)
q75_top_confidence_incorrect = percentile(incorrect_classification_top_confidences, 75)
iqr_top_confidence_incorrect = q75_top_confidence_incorrect - q25_top_confidence_incorrect

avg_margin_correct = average(correct_classification_margins)
std_margin_correct = std(correct_classification_margins)
median_margin_correct = median(correct_classification_margins)
q25_margin_correct = percentile(correct_classification_margins, 25)
q75_margin_correct = percentile(correct_classification_margins, 75)
iqr_margin_correct = q75_margin_correct - q25_margin_correct

avg_margin_incorrect = average(incorrect_classification_margins)
std_margin_incorrect = std(incorrect_classification_margins)
median_margin_incorrect = median(incorrect_classification_margins)
q25_margin_incorrect = percentile(incorrect_classification_margins, 25)
q75_margin_incorrect = percentile(incorrect_classification_margins, 75)
iqr_margin_incorrect = q75_margin_incorrect - q25_margin_incorrect

print(f"Top {TOP_K} Stats")
print(f"avg_top_confidence_correct: \n{avg_top_confidence_correct}")
print(f"median_top_confidence_correct: \n{median_top_confidence_correct}")
print(f"iqr_top_confidence_correct: \n{iqr_top_confidence_correct} (Q25={q25_top_confidence_correct}, Q75={q75_top_confidence_correct})")
print(f"avg_top_confidence_incorrect: \n{avg_top_confidence_incorrect}")
print(f"median_top_confidence_incorrect: \n{median_top_confidence_incorrect}")
print(f"iqr_top_confidence_incorrect: \n{iqr_top_confidence_incorrect} (Q25={q25_top_confidence_incorrect}, Q75={q75_top_confidence_incorrect})")
print(f"avg_margin_correct: \n{avg_margin_correct}")
print(f"median_margin_correct: \n{median_margin_correct}")
print(f"iqr_margin_correct: \n{iqr_margin_correct} (Q25={q25_margin_correct}, Q75={q75_margin_correct})")
print(f"avg_margin_incorrect: \n{avg_margin_incorrect}")
print(f"median_margin_incorrect: \n{median_margin_incorrect}")
print(f"iqr_margin_incorrect: \n{iqr_margin_incorrect} (Q25={q25_margin_incorrect}, Q75={q75_margin_incorrect})")
print(f"Total Times Top Classification was Incorrect: \n{count_incorrect_top_classification}")
print(f"Times Top Classification was Incorrect but True classification was in top {TOP_K}: \n{count_incorrect_class_but_in_top_k}")

margins = np.array([r[0] for r in records])
top1_correct = np.array([r[2] for r in records])
total_wrong = (~top1_correct).sum()

for t in np.arange(0.1, 1.0, 0.1):
    send = margins < t
    sent = send.sum()
    sent_and_wrong = (send & ~top1_correct).sum()
    precision = sent_and_wrong / sent if sent else 0
    recall = sent_and_wrong / total_wrong
    print(f"t={t:.2f}  sent={sent:5d}  precision={precision:.3f}  recall={recall:.3f}")

    # Classify image and get top 5
    #1. Check if true classification is in top-5
    #2. Check if confidence threshold is reached
    #3.

    # if images have low or similar confidence are they likely to have the correct classification in the list of top 5?
    # start with average confidence. What is average confidence of correct classification vs incorrect classification

# does true label match?
# I have to be CAREFUL! Claude is probably smart enough to look at the image filename or path to learn its class.
# subset I am interested in are images whose top level classification is incorrect, but the true classification is in
# the top 5.
# Of all images classified incorrectly, how many have the potential to be reclassified? This is the subset I am
# interested in, and how big is it relative to incorrect classifications?
# if this is too low, cound expand to top 10

# Ongoing Questions:
# how do I compare acuracy? Maybe just top class accuracy? Is there a way to still do top 5? I;d have to
# manually make the metric myself which would be really annoying. Basically i'd just swap the order
# if the VLM decides to, and then return the same top5 tensor, and let the external method check the classification.

# What if we LOOK at top 10 for the VLM rescue, but still only return the top5? That might work

# parameterized return of top 5