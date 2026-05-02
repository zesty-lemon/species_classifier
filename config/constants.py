from pathlib import Path

# Absolute project root — works regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Two places to look for large datasets (training images, validation images, annotations)
# Priority 1: /data directory one level above the project
LOCAL_DATA_DIR = PROJECT_ROOT.parent / "data"
# Priority 2: /data directory on removable volume
EXTERNAL_DATA_DIR = Path("/Volumes/giDrive/data")

# Project-internal data (shapefiles, mini dataset annotations — always in the repo)
INTERNAL_DATA_DIR = PROJECT_ROOT / "data"

# Dataset version names
MINI_DATASET = "2021_train_mini"
FULL_DATASET = "2021_train"
VAL_DATASET = "2021_valid"


def resolve_data_dir(dataset_name: str) -> Path:
    """Find the data directory containing the given dataset.
    Checks local first, then the external volume."""
    for base in (LOCAL_DATA_DIR, EXTERNAL_DATA_DIR):
        candidate = base / dataset_name
        if candidate.is_dir():
            print(f"Found dataset '{dataset_name}' at {candidate}")
            return base
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found in:\n"
        f"  {LOCAL_DATA_DIR}\n"
        f"  {EXTERNAL_DATA_DIR}\n"
        f"Check that the data exists in one of these locations."
    )

# Static instructions — passed as the `system` parameter to the Messages API.
VLM_SYSTEM_PROMPT = (
    "You are the tie-breaker for a ResNet-50 plant species classifier. The model's top-K candidates for "
    "this image have a low confidence margin, so you are being asked to pick the best match from them. "
    "All images are of species found in Vermont.\n\n"
    "Label format: <5-digit_class_id>_<Kingdom>_<Phylum>_<Class>_<Order>_<Family>_<Genus>_<species>\n\n"
    "Rules:\n"
    "1. You MUST pick exactly one species from the list provided in the user message. Do not suggest any "
    "species not in the list, even if the image appears to show something else. If nothing looks right, "
    "return the option that is the closest visual match.\n"
    "2. Respond with ONLY the full class label, copied verbatim from the list (including the 5-digit prefix). "
    "Do not include reasoning, explanation, or any other text."
)

# Header prefixed to the dynamic candidate list in the user message.
VLM_CANDIDATES_HEADER = "Candidate species (ordered by ResNet confidence, highest first):\n"
