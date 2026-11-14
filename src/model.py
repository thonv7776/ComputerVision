"""Small CNN: 3 x (Conv + ReLU) with max pooling, global average pooling, linear classifier."""
from torch import nn


class FruitCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 3x128x128 -> 16x64x64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # -> 32x32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),                    # -> 64x32x32
            nn.AdaptiveAvgPool2d(1))                                       # -> 64x1x1
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))   # logits; CrossEntropyLoss applies softmax itself
