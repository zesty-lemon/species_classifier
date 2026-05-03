import os
# Force any unsupported MPS op to raise instead of silently falling back to CPU.
# Silent CPU fallbacks cause per-batch GPU<->CPU tensor shuttling that destroys throughput.
# Must be set before `import torch`.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch

"""
Reusable configurations for training. Use the GPU if available, otherwise use MPS (Mac GPU). 
If neither available fallback to CPU only
"""
torch.manual_seed(42)
np.random.seed(42)

# Configure the device to use GPU (cuda) if available, otherwise MPS (mac) if available, otherwise fallback to CPU
device_name = 'cpu'
if torch.cuda.is_available():
    device_name = 'cuda'
elif torch.backends.mps.is_available():
    device_name = 'mps'

device = torch.device(device_name)
print(f"Using device: {device}")
