"""Transformer-based text encoders and classifiers."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from utils.config import TextTransformerConfig, NUM_CLASSES


class TransformerTextEncoder(nn.Module):
    def __init__(self, cfg: TextTransformerConfig):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        hidden_size = self.backbone.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, cfg.feature_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.feature_dim),
        )

        self.out_dim = cfg.feature_dim

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, text: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=text, attention_mask=attention_mask)
        cls_feat = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_feat)


class TextOnlyTransformerModel(nn.Module):
    def __init__(self, cfg: TextTransformerConfig, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = TransformerTextEncoder(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.feature_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.feature_dim, num_classes),
        )

    def forward(self, text: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.encoder(text=text, attention_mask=attention_mask)
        return self.classifier(features)