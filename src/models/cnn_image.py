"""
models/cnn_image.py
─────────────────────────────────────────────────────────────────────────────
CNN facial feature extractor for emotion recognition.

Architecture:
  3× Conv block (Conv → BN → ReLU → MaxPool → Dropout)
  → Global Average Pooling
  → FC head

Designed for 48×48 grayscale images (FER2013).
For color images, set input_channels=3 in ImageCNNConfig.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import ImageCNNConfig, NUM_CLASSES


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageCNNEncoder(nn.Module):
    """
    Lightweight CNN that outputs a fixed-size feature vector per image.
    Used as a visual branch inside MultimodalFusion.

    Input  : (B, C, 48, 48)
    Output : (B, feature_dim)
    """

    def __init__(self, cfg: ImageCNNConfig):
        super().__init__()
        self.feature_dim = cfg.feature_dim
        C = cfg.input_channels

        self.conv_layers = nn.Sequential(
            ConvBlock(C,   32, dropout=0.20),    # → (B, 32, 24, 24)
            ConvBlock(32,  64, dropout=0.25),    # → (B, 64, 12, 12)
            ConvBlock(64, 128, dropout=0.25),    # → (B, 128, 6, 6)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, cfg.feature_dim),
            nn.ReLU(),
            nn.LayerNorm(cfg.feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_layers(x)          # (B, 128, 6, 6)
        pooled = self.global_avg_pool(feat) # (B, 128, 1, 1)
        return self.projection(pooled)      # (B, feature_dim)


class ImageOnlyModel(nn.Module):
    """
    Full classification model for image-only emotion detection.
    Wraps ImageCNNEncoder with a classification head.
    """

    def __init__(self, cfg: ImageCNNConfig, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = ImageCNNEncoder(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        features = self.encoder(image)
        return self.classifier(features)    # (B, num_classes)
