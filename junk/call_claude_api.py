import base64

from utils.dataset_utils import read_image_annotations_from_file

from dotenv import load_dotenv
load_dotenv() #loads the api key from .env file
from config import constants as c

import anthropic
client = anthropic.Anthropic()
# IMAGE_PATH = "/Users/giles/Documents/Grad_School/Spring_2026/deep_learning/Project/data/2021_valid/05729_Plantae_Bryophyta_Bryopsida_Bryales_Bryaceae_Bryum_argenteum/0d2440a5-cddf-4883-9b04-03d31dfba70d.jpg"
IMAGE_PATH = "/Users/giles/Library/CloudStorage/GoogleDrive-gileslemmon@gmail.com/My Drive/deep_learning/data/2021_valid/09346_Plantae_Tracheophyta_Magnoliopsida_Rosales_Rhamnaceae_Rhamnus_humboldtiana/1c31a45a-7e99-4568-b760-f127e07313c7.jpg"
DATASET_USED = c.FULL_DATASET

with open(IMAGE_PATH, "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

species_top_5_classes = ["09318_Plantae_Tracheophyta_Magnoliopsida_Rosales_Rhamnaceae_Ceanothus_americanus",
                 "08835_Plantae_Tracheophyta_Magnoliopsida_Malpighiales_Hypericaceae_Hypericum_punctatum",
                 "08998_Plantae_Tracheophyta_Magnoliopsida_Myrtales_Lythraceae_Decodon_verticillatus",
                 "09684_Plantae_Tracheophyta_Magnoliopsida_Solanales_Convolvulaceae_Cuscuta_gronovii",
                 "09494_Plantae_Tracheophyta_Magnoliopsida_Rosales_Rosaceae_Spiraea_japonica"]

species_probs = [0.3214,
                 0.2895,
                 0.2794,
                 0.0428,
                 0.0121]

# todo: include gps coordinates or maybe location
# annotations = read_image_annotations_from_file(dataset_name=DATASET_USED)


prompt = ("You are the final layer of an image classification architecture. A resnet50 model has examined an image"
          "and identified the top possible classes and their probabilities. However - the model has a low margin"
          "between these classes. It is your job to examine the image, and decide which of the top classes"
          "best fits the image. The classes are the names of the possible species. All images are of species"
          "found in Vermont. The syntax for the species names is <5-digit_class_id>_<Kingdom>_<Phylum>_<Class>_<Order>_<Family>_<Genus>_<species>"
          "I want your output to be the full class label. I want you to pick the class (species) that best describes the image"
          "and give me the full class label in your output. "
          "Here are the potential top classes and their confidences: ")

prompt = prompt + ", ".join(species_top_5_classes)

message = client.messages.create(
    model="claude-opus-4-7",
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
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ],
)

vlm_picked_class = ""
for block in message.content:
    if block.type == "text":
        print(block.text)
        vlm_picked_class = block.text
print(f"VLM Picked Class: {vlm_picked_class}")

for block in message.content:
    if block.type == "text":
        print(block.text)

