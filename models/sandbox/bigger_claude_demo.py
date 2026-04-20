"""
Take a model that has already been trained and asses its performance
"""
import anthropic
import torch
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv() #loads the api key from .env file
from config import constants as c
import base64
import os

from config.device_config import device, device_name
import utils.data_load_and_config_util as data_config
from config import constants as c
from model_utils.model_utils import get_trained_model, persist_trained_model

# ------------ Initial Configuration ------------
MODEL_NAME = "ResNet50 Scratch Trained"
MODEL_PATH = "/Users/giles/Documents/Grad_School/Spring_2026/deep_learning/Project/species_classifier/models/trained_models/ResNet50_Scratch_Trained/2021_train/ResNet50_Scratch_Trained_model.joblib"
DATASET_USED = c.FULL_DATASET # Need to leave this even though it isn't being used, need it for category indexes
TOP_K = 5
MARGIN_THRESHOLD = 0.5  # Only send to VLM when ResNet's margin < this value


def find_claude_pick(response_text, candidate_names):
    """Return the index into candidate_names that Claude picked, or None if no match."""
    lowered = response_text.lower()
    for pos, name in enumerate(candidate_names):
        if name.lower() in lowered:
            return pos
    # Fall back to the 5-digit class ID prefix (unique within top-K)
    for pos, name in enumerate(candidate_names):
        class_id = name.split("_", 1)[0]
        if class_id in response_text:
            return pos
    return None


def get_message(encoded_image_data, final_prompt):
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": final_prompt,
                    },
                ],
            }
        ],
    )
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


baseline_correct = 0
post_vlm_correct = 0
vlm_calls = 0
total_evaluated = 0

client = anthropic.Anthropic()
label_to_category = val_loader.dataset.label_to_category

flat_dataset = val_loader.dataset
base_dataset = flat_dataset.base_dataset

i = 0
for image, label in tqdm(flat_dataset, desc="Evaluating", unit=" images"):
    input_tensor = image.unsqueeze(0).to(device)

    real_idx = flat_dataset.indices[i]
    cat_id, filename = base_dataset.index[real_idx]
    image_path = os.path.join(base_dataset.root, base_dataset.all_categories[cat_id], filename)
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    with torch.no_grad():
        logits = trained_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        topk_probs, topk_idx = probs.topk(TOP_K, dim=1)

        topk_probs_list = topk_probs.tolist()[0]
        topk_idx_list = topk_idx.tolist()[0]
        margin = topk_probs_list[0] - topk_probs_list[1]

        baseline_top1_correct = (topk_idx_list[0] == label)

        # Only rescue with the VLM when ResNet's margin is below threshold.
        # Above threshold, we trust ResNet's top-1 and skip the API call entirely.
        if margin < MARGIN_THRESHOLD:
            species_names = [label_to_category[topk_idx_list[idx]] for idx in range(len(topk_idx_list))]

            species_lines = [
                f"- {name} (confidence: {prob:.3f})"
                for name, prob in zip(species_names, topk_probs_list)
            ]
            prompt = c.VLM_BASE_PROMPT + "\n".join(species_lines)
            message = get_message(image_data, prompt)
            vlm_calls += 1

            vlm_picked_class = ""
            for block in message.content:
                if block.type == "text":
                    print(block.text)
                    vlm_picked_class = block.text

            # Match Claude's response to one of the top-K species, then swap
            # that entry to rank 0 so downstream top-1 accuracy reflects the VLM pick.
            claude_pick_pos = find_claude_pick(vlm_picked_class, species_names)
            if claude_pick_pos is not None and claude_pick_pos != 0:
                topk_idx_list[0], topk_idx_list[claude_pick_pos] = topk_idx_list[claude_pick_pos], topk_idx_list[0]
                topk_probs_list[0], topk_probs_list[claude_pick_pos] = topk_probs_list[claude_pick_pos], topk_probs_list[0]

        post_vlm_top1_correct = (topk_idx_list[0] == label)

        if baseline_top1_correct:
            baseline_correct += 1
        if post_vlm_top1_correct:
            post_vlm_correct += 1
        total_evaluated += 1

    # only do 200 transactions as a test
    i = i+1
    if (i>200):
        break

baseline_acc = baseline_correct / total_evaluated
post_vlm_acc = post_vlm_correct / total_evaluated
delta = post_vlm_acc - baseline_acc

print(f"------- VLM Rescue Experiment -------")
print(f"\nEvaluated: {total_evaluated} images")
print(f"VLM rescue calls: {vlm_calls} ({vlm_calls/total_evaluated:.1%} of images)")
print(f"Baseline ResNet top-1: {baseline_acc:.3%} ({baseline_correct}/{total_evaluated})")
print(f"Post-VLM top-1:        {post_vlm_acc:.3%} ({post_vlm_correct}/{total_evaluated})")
print(f"Delta:                 {delta:+.3%}")

