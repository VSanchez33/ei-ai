"""Text tokenization and image transform utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List
from torchvision import transforms

import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import AutoTokenizer


class HFTextTokenizer:
    """Thin wrapper around a Hugging Face tokenizer used by the project."""

    def __init__(self, model_name: str = "distilbert-base-uncased", max_seq_len: int = 64):
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

    def encode_one(self, text: str) -> Dict[str, List[int]]:
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_attention_mask=True,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    def save(self, path: str) -> None:
        payload = {"model_name": self.model_name, "max_seq_len": self.max_seq_len}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "HFTextTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(model_name=payload["model_name"], max_seq_len=payload["max_seq_len"])


class SimpleImageTransform:
    def __init__(self, image_size: int = 112, train: bool = False):
        self.image_size = image_size
        self.train = train
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((self.image_size, self.image_size))

        if self.train:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            angle = random.uniform(-8, 8)
            img = img.rotate(angle)
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = (arr - self.mean) / self.std
        return torch.tensor(arr, dtype=torch.float32)


#def get_image_transforms(image_size: int = 112, split: str = "train"):
    #return SimpleImageTransform(image_size=image_size, train=(split == "train"))

def get_image_transforms(image_size=224, split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def preprocess_image_array(arr: np.ndarray, image_size: int = 112) -> torch.Tensor:
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    tf = get_image_transforms(image_size, split="test")
    return tf(img).unsqueeze(0)
