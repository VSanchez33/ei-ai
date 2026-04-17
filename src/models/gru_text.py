"""
models/gru_text.py
─────────────────────────────────────────────────────────────────────────────
Bidirectional GRU encoder for text emotion recognition.

Architecture:
  Embedding → Dropout → BiGRU (N layers) → Attention Pool → Linear head

The GRU was chosen over LSTM for:
  • Fewer parameters (2 gates vs 3) → faster training
  • Similar or better performance on short-to-medium sequences
  • Good inductive bias for sequential language data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import TextGRUConfig, NUM_CLASSES


class AttentionPool(nn.Module):
    """
    Soft attention over GRU hidden states.
    Learns which timesteps are most informative for the emotion label.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        hidden : (B, T, H)
        mask   : (B, T) — 1 for valid tokens, 0 for padding
        returns: (B, H) weighted context vector
        """
        scores = self.attn(hidden).squeeze(-1)          # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)             # (B, T)
        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)  # (B, H)
        return context


class TextGRUEncoder(nn.Module):
    """
    Standalone text emotion encoder.
    Can be used as a submodule inside MultimodalFusion or independently.
    """

    def __init__(self, cfg: TextGRUConfig):
        super().__init__()
        self.cfg = cfg
        out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)

        self.embedding = nn.Embedding(
            cfg.vocab_size, cfg.embed_dim, padding_idx=0
        )
        self.embed_drop = nn.Dropout(cfg.dropout)

        self.gru = nn.GRU(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )

        self.attn_pool = AttentionPool(out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.embed_drop(self.embedding(tokens))
        hidden, h_n = self.gru(emb)

        if self.cfg.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last_hidden = h_n[-1]

        last_hidden = self.layer_norm(last_hidden)
        return last_hidden


class TextOnlyModel(nn.Module):
    """
    Full classification model for text-only emotion detection.
    Wraps TextGRUEncoder with a classification head.
    """

    def __init__(self, cfg: TextGRUConfig, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = TextGRUEncoder(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        tokens: torch.Tensor = None,
        lengths: torch.Tensor = None,
        text: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        tokens = tokens if tokens is not None else text
        lengths = lengths if lengths is not None else text_lengths
        if tokens is None or lengths is None:
            raise ValueError("TextOnlyModel requires token ids and lengths.")
        features = self.encoder(tokens, lengths)
        return self.classifier(features)
