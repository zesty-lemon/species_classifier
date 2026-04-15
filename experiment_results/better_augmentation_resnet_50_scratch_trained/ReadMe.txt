Baseline Model

Resnet50 Scratch Trained

follows the standard architecture from He et al. (2015) with a bottleneck block design

ResNet-50 with bottleneck blocks ([3, 4, 6, 3]), expansion factor 4. Stem: 7x7 conv (stride 2) → BN → ReLU → 3x3 max pool (stride 2). Each bottleneck: 1x1 → 3x3 → 1x1 with BN/ReLU and identity or projection
  shortcut. Stage outputs: 256 → 512 → 1024 → 2048 channels. Global average pool → FC to N classes.

Training: SGD, fixed lr=0.01, batch size 128, 10 epochs, cross-entropy loss.

Augmentation:
Improved from baseline model
RandomResizedCrop to 224x224, random horizontal flip, random rotation (±15°), color jitter (brightness=0.2, contrast=0.2, saturation=0.2), ImageNet normalization. Validation uses deterministic
  resize to 224x224 with ImageNet normalization only.