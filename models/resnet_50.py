import torch
import torch.nn as nn

class ResNet50_Model(nn.Module):
    """
    ResNet-50 with Bottleneck blocks. Configuration: [3, 4, 6, 3]
    """

    def __init__(self, num_classes=10):
        super(ResNet50_Model, self).__init__()

        # ===== 1. Initial Convolution (Stem) =====
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ===== 2. Max Pooling =====
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ===== 3. Residual Stages =====
        # ResNet-50 configuration: [3, 4, 6, 3] blocks per stage
        self.layer1 = self._make_layer(in_channels=64, channels=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(in_channels=256, channels=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(in_channels=512, channels=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(in_channels=1024, channels=512, blocks=3, stride=2)

        # ===== 4. Classification Head =====
        # Global Average Pooling: reduces any spatial size to 1x1
        # This replaces heavy FC layers (like in VGG) and adds translation invariance
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Final classifier: 512 features -> num_classes
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, channels, blocks, stride=1):
        """
        Create a layer with BottleneckBlock_Exercise blocks.
        First block uses stride, rest use stride=1.
        """
        layers = []
        # First block: handles channel change and optional downsampling
        layers.append(BottleNeck_Block(in_channels=in_channels, channels=channels, stride=stride))

        # Remaining blocks: input channels = output channels of previous block (channels * expansion)
        for _ in range(1, blocks):
            layers.append(BottleNeck_Block(in_channels=channels * BottleNeck_Block.expansion,
                                           channels=channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem: Initial feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Four residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling + flatten
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc(x)
        return x

class BottleNeck_Block(nn.Module):
    """
    Bottleneck Block for ResNet-50.
    Structure: Conv1x1(reduce) -> Conv3x3 -> Conv1x1(expand) -> skip connection
    """
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, expansion=expansion):
        super(BottleNeck_Block, self).__init__()
        # --- Convolution Layers ---
        # Conv1: 1x1
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Conv2: 3x3, with stride parameter
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # Conv3: 1x1, expand to channels * expansion
        self.conv3 = nn.Conv2d(channels, channels * expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * expansion)

        # --- ReLU Layer ---
        self.relu = nn.ReLU(inplace=True)

        # Default: Skip Connection (identity)
        self.shortcut = nn.Identity()

        # 1x1 conv + bn if dimensions change
        # Adjust shortcut if dimensions change
        if stride != 1 or in_channels != channels * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * expansion)
            )

    # Forward Pass
    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out