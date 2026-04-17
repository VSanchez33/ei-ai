"""
models/audio_stub.py
─────────────────────────────────────────────────────────────────────────────
Audio/Speech Emotion Recognition — Future Expansion Module

This stub defines the interface that an audio GRU encoder must satisfy.
When you're ready to add speech support:

  1. Install: pip install librosa soundfile torchaudio
  2. Implement AudioGRUEncoder.forward() using MFCC or mel-spectrogram input
  3. Update MultimodalFusion to accept audio tensors
  4. Update dataset.py to load and preprocess audio files

INPUT FORMAT (expected)
  Tensor of shape (B, T, n_mfcc) — batch of MFCC sequences
  where T = time frames, n_mfcc = number of mel-frequency cepstral coefficients

RECOMMENDED APPROACH
  • Extract MFCCs with librosa (n_mfcc=40, hop_length=512)
  • Normalize per utterance
  • Feed into a BiGRU (same pattern as TextGRUEncoder)
  • Optionally: use pre-trained wav2vec 2.0 (facebook/wav2vec2-base) as encoder

EXAMPLE PREPROCESSING
─────────────────────
    import librosa
    import numpy as np

    def extract_mfcc(audio_path: str, sr: int = 22050, n_mfcc: int = 40) -> np.ndarray:
        y, sr = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)   # (n_mfcc, T)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)        # normalize
        return mfcc.T  # (T, n_mfcc) — time-first for GRU

DATASETS FOR AUDIO EMOTION
  • RAVDESS  — https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
  • CREMAD   — https://www.kaggle.com/datasets/ejlok1/cremad
  • IEMOCAP  — https://sail.usc.edu/iemocap/
"""

import torch
import torch.nn as nn
from utils.config import AudioGRUConfig


class AudioGRUEncoder(nn.Module):
    """
    Stub audio encoder. Replace the forward() body with real MFCC-based GRU.

    Input  : (B, T, n_mfcc)
    Output : (B, feature_dim)
    """

    def __init__(self, cfg: AudioGRUConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_dim = cfg.feature_dim
        self._implemented = False

        # ── Architecture (to be activated when you implement this) ────────────
        self.input_norm = nn.LayerNorm(cfg.n_mfcc)
        self.gru = nn.GRU(
            input_size=cfg.n_mfcc,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.feature_dim),
            nn.ReLU(),
            nn.LayerNorm(cfg.feature_dim),
        )

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mfcc: (B, T, n_mfcc) — padded MFCC sequences

        Returns:
            (B, feature_dim) — audio emotion embedding

        NOTE: When implementing, also accept `lengths` tensor for pack_padded_sequence.
        """
        # ── STUB: returns zeros until implemented ─────────────────────────────
        B = mfcc.size(0)
        return torch.zeros(B, self.feature_dim, device=mfcc.device)

        # ── REAL IMPLEMENTATION (uncomment & complete when ready) ─────────────
        # normed = self.input_norm(mfcc)
        # packed = nn.utils.rnn.pack_padded_sequence(normed, lengths.cpu(),
        #                                            batch_first=True, enforce_sorted=False)
        # packed_out, _ = self.gru(packed)
        # hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # # Mean pool over time
        # context = hidden.mean(dim=1)              # (B, 2*hidden_dim)
        # return self.projection(context)           # (B, feature_dim)
