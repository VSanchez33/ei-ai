"""ResNet-based image encoder for image-only and multimodal emotion recognition."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ImageCNNEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Freeze most ResNet layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Fine-tune final ResNet block
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        self.projection = nn.Sequential(
            nn.Linear(in_features, cfg.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(cfg.feature_dim),
        )

        self.out_dim = cfg.feature_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        features = self.projection(features)
        return features


class ImageOnlyModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.encoder = ImageCNNEncoder(cfg)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.feature_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.encoder(image)
        return self.classifier(features)