"""Centralized configuration for the emotion detection system."""

from dataclasses import dataclass, field
from typing import List

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
NUM_CLASSES = len(EMOTIONS)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}


@dataclass
class TextTransformerConfig:
    model_name: str = "distilbert-base-uncased"
    max_seq_len: int = 64
    feature_dim: int = 256
    dropout: float = 0.2
    freeze_backbone: bool = False


@dataclass
class ImageCNNConfig:
    input_channels: int = 3
    image_size: int = 224
    feature_dim: int = 256


@dataclass
class AudioGRUConfig:
    n_mfcc: int = 40
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    feature_dim: int = 128


@dataclass
class FusionConfig:
    fusion_hidden: int = 256
    dropout: float = 0.3
    use_attention: bool = True


@dataclass
class TrainConfig:
    modality: str = "multimodal"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    patience: int = 4
    seed: int = 42
    num_workers: int = 2
    checkpoint_dir: str = "checkpoints"
    dataset_type: str = "meld"


@dataclass
class Config:
    text: TextTransformerConfig = field(default_factory=TextTransformerConfig)
    image: ImageCNNConfig = field(default_factory=ImageCNNConfig)
    audio: AudioGRUConfig = field(default_factory=AudioGRUConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


cfg = Config()
