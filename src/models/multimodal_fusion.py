"""Multimodal fusion models using a CNN image branch and Transformer text branch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import TextTransformerConfig, ImageCNNConfig, AudioGRUConfig, FusionConfig, NUM_CLASSES
from models.transformer_text import TransformerTextEncoder, TextOnlyTransformerModel
from models.cnn_image import ImageCNNEncoder, ImageOnlyModel
from models.audio_stub import AudioGRUEncoder


# =========================
# Cross-modal attention
# =========================
class CrossModalAttention(nn.Module):
    def __init__(self, dim_q: int, dim_k: int, hidden: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, hidden, bias=False)
        self.k_proj = nn.Linear(dim_k, hidden, bias=False)
        self.v_proj = nn.Linear(dim_k, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, dim_q)
        self.scale = hidden ** -0.5

    def forward(self, query: torch.Tensor, key_val: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query).unsqueeze(1)
        k = self.k_proj(key_val).unsqueeze(1)
        v = self.v_proj(key_val).unsqueeze(1)

        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        attended = torch.bmm(attn, v).squeeze(1)

        return query + self.out_proj(attended)


# =========================
# Multimodal Model
# =========================
class MultimodalEmotionModel(nn.Module):
    def __init__(
        self,
        text_cfg: TextTransformerConfig,
        image_cfg: ImageCNNConfig,
        audio_cfg: AudioGRUConfig,
        fusion_cfg: FusionConfig,
        num_classes: int = NUM_CLASSES,
        use_audio: bool = False,
    ):
        super().__init__()

        self.use_audio = use_audio
        self.use_attention = fusion_cfg.use_attention

        self.text_encoder = TransformerTextEncoder(text_cfg)
        self.image_encoder = ImageCNNEncoder(image_cfg)

        if use_audio:
            self.audio_encoder = AudioGRUEncoder(audio_cfg)

        text_dim = self.text_encoder.out_dim
        image_dim = image_cfg.feature_dim
        audio_dim = audio_cfg.feature_dim if use_audio else 0

        # Attention
        if self.use_attention:
            self.img2txt_attn = CrossModalAttention(image_dim, text_dim)
            self.txt2img_attn = CrossModalAttention(text_dim, image_dim)

        # Fusion
        fused_dim = text_dim + image_dim + audio_dim

        self.fusion_dropout = nn.Dropout(fusion_cfg.dropout)

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_cfg.fusion_hidden),
            nn.GELU(),
            nn.Dropout(fusion_cfg.dropout),
            nn.Linear(fusion_cfg.fusion_hidden, fusion_cfg.fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(fusion_cfg.dropout),
            nn.Linear(fusion_cfg.fusion_hidden // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        attention_mask: torch.Tensor,
        audio: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        # Encode
        img_feat = self.image_encoder(image)
        txt_feat = self.text_encoder(text=text, attention_mask=attention_mask)

        # Cross-attention
        if self.use_attention:
            img_feat = self.img2txt_attn(img_feat, txt_feat)
            txt_feat = self.txt2img_attn(txt_feat, img_feat)

        # Collect features
        modality_feats = [img_feat, txt_feat]

        if self.use_audio and audio is not None:
            modality_feats.append(self.audio_encoder(audio))

        # Fusion
        fused = torch.cat(modality_feats, dim=-1)
        fused = self.fusion_dropout(fused)

        return self.fusion_head(fused)


# =========================
# Factory
# =========================
def build_model(
    modality: str,
    text_cfg: TextTransformerConfig = None,
    image_cfg: ImageCNNConfig = None,
    audio_cfg: AudioGRUConfig = None,
    fusion_cfg: FusionConfig = None,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    from utils.config import cfg as default_cfg

    text_cfg = text_cfg or default_cfg.text
    image_cfg = image_cfg or default_cfg.image
    audio_cfg = audio_cfg or default_cfg.audio
    fusion_cfg = fusion_cfg or default_cfg.fusion

    if modality == "text":
        return TextOnlyTransformerModel(text_cfg, num_classes)

    if modality == "image":
        return ImageOnlyModel(image_cfg, num_classes)

    if modality == "multimodal":
        return MultimodalEmotionModel(text_cfg, image_cfg, audio_cfg, fusion_cfg, num_classes, use_audio=False)

    if modality == "audio":
        return MultimodalEmotionModel(text_cfg, image_cfg, audio_cfg, fusion_cfg, num_classes, use_audio=True)

    raise ValueError(f"Unknown modality: {modality!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)