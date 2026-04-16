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
