from __future__ import annotations

"""MELD dataset utilities for image, text, and multimodal emotion recognition."""

import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import HFTextTokenizer, get_image_transforms
from utils.config import cfg

MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MELD_ALIASES = {
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "happy": "joy",
    "neutral": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
}


@dataclass
class Sample:
    image_path: Optional[str]
    text: Optional[str]
    label_name: str
    label_idx: int
    split: str
    dialogue_id: Optional[int] = None
    utterance_id: Optional[int] = None


class EmotionMultimodalDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        modality: str,
        tokenizer: Optional[HFTextTokenizer],
        image_transform,
    ):
        self.samples = list(samples)
        self.modality = modality
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.label_names = sorted({s.label_name for s in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        item: Dict[str, torch.Tensor] = {"label": torch.tensor(sample.label_idx, dtype=torch.long)}

        if self.modality in {"image", "multimodal", "audio"}:
            if not sample.image_path:
                raise ValueError("Image path missing for image/multimodal sample.")
            img = Image.open(sample.image_path).convert("RGB")
            item["image"] = self.image_transform(img)

        if self.modality in {"text", "multimodal", "audio"}:
            if not sample.text:
                raise ValueError("Text missing for text/multimodal sample.")
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for text-containing modalities.")
            enc = self.tokenizer.encode_one(sample.text)
            item["text"] = torch.tensor(enc["input_ids"], dtype=torch.long)
            item["attention_mask"] = torch.tensor(enc["attention_mask"], dtype=torch.long)

        return item


def _canonicalize_meld_label(label: str) -> Optional[str]:
    if pd.isna(label):
        return None
    cleaned = str(label).strip().lower()
    return MELD_ALIASES.get(cleaned)



def _frame_candidates(frame_root: Path, split: str, dialogue_id: int, utterance_id: int) -> List[Path]:
    stem = f"dia{dialogue_id}_utt{utterance_id}"
    return [
        frame_root / split / stem / "frame.jpg",
        frame_root / split / f"{stem}.jpg",
        frame_root / stem / "frame.jpg",
        frame_root / f"{stem}.jpg",
    ]



def _resolve_visual_path(meld_root: Path, frame_root: Path, split: str, dialogue_id: int, utterance_id: int) -> Optional[Path]:
    for candidate in _frame_candidates(frame_root, split, dialogue_id, utterance_id):
        if candidate.exists():
            return candidate
    return None

    video_root = _infer_split_dir(meld_root, split)
    for candidate in _video_candidates(video_root, dialogue_id, utterance_id):
        if candidate.exists():
            return candidate
    return None


def _load_split_csv(meld_root: Path, split: str) -> pd.DataFrame:
    csv_map = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}
    csv_path = meld_root / csv_map[split]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing MELD annotation file: {csv_path}")
    return pd.read_csv(csv_path)


def _ensure_tar_present(meld_root: Path) -> None:
    tar_path = meld_root / "MELD.Raw.tar.gz"
    if not tar_path.exists():
        return
    # informational only; extraction is handled by prepare_meld.py
    return


def build_meld_samples(meld_root: str, frame_root: Optional[str], modality: str) -> Tuple[Dict[str, List[Sample]], List[str], Optional[HFTextTokenizer]]:
    meld_root_p = Path(meld_root)
    _ensure_tar_present(meld_root_p)
    frame_root_p = Path(frame_root) if frame_root else meld_root_p / "frames"

    split_samples: Dict[str, List[Sample]] = {"train": [], "dev": [], "test": []}
    observed_labels = set()

    for split in ["train", "dev", "test"]:
        df = _load_split_csv(meld_root_p, split)
        for _, row in df.iterrows():
            label_name = _canonicalize_meld_label(row.get("Emotion"))
            text = str(row.get("Utterance", "")).strip()
            dialogue_id = int(row.get("Dialogue_ID"))
            utterance_id = int(row.get("Utterance_ID"))
            if not label_name or not text:
                continue

            visual_path = _resolve_visual_path(
                meld_root_p, frame_root_p, split, dialogue_id, utterance_id
            )
            if modality in {"image", "multimodal", "audio"}:
                if visual_path is None:
                # Skip samples whose extracted frame is missing
                    continue

            observed_labels.add(label_name)
            split_samples[split].append(
                Sample(
                    image_path=str(visual_path) if visual_path else None,
                    text=text,
                    label_name=label_name,
                    label_idx=-1,
                    split=split,
                    dialogue_id=dialogue_id,
                    utterance_id=utterance_id,
                )
            )

    label_names = [lbl for lbl in MELD_EMOTIONS if lbl in observed_labels]
    if len(label_names) < 2:
        raise ValueError(
            "Not enough MELD classes found. Make sure the CSV files are present and frames/videos are prepared."
        )

    label_to_idx = {lbl: i for i, lbl in enumerate(label_names)}
    for split in split_samples:
        split_samples[split] = [
            Sample(
                image_path=s.image_path,
                text=s.text,
                label_name=s.label_name,
                label_idx=label_to_idx[s.label_name],
                split=s.split,
                dialogue_id=s.dialogue_id,
                utterance_id=s.utterance_id,
            )
            for s in split_samples[split]
            if s.label_name in label_to_idx
        ]

    tokenizer = None
    if modality in {"text", "multimodal", "audio"}:
        tokenizer = HFTextTokenizer(model_name=cfg.text.model_name, max_seq_len=cfg.text.max_seq_len)

    return split_samples, label_names, tokenizer


class DatasetBundle:
    def __init__(self, train_ds, val_ds, test_ds, tokenizer, label_names):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.tokenizer = tokenizer
        self.label_names = label_names


def build_datasets(
    image_root: str,
    text_csv: str,
    modality: str = "multimodal",
    seed: int = 42,
    dataset_type: str = "meld",
    meld_root: str = "dataset/meld",
    frame_root: Optional[str] = None,
) -> DatasetBundle:
    if dataset_type != "meld":
        raise ValueError("This updated pipeline is MELD-first. Use --dataset_type meld.")

    split_samples, label_names, tokenizer = build_meld_samples(meld_root=meld_root, frame_root=frame_root, modality=modality)

    train_ds = EmotionMultimodalDataset(
        split_samples["train"], modality=modality, tokenizer=tokenizer,
        image_transform=get_image_transforms(cfg.image.image_size, split="train"),
    )
    val_ds = EmotionMultimodalDataset(
        split_samples["dev"], modality=modality, tokenizer=tokenizer,
        image_transform=get_image_transforms(cfg.image.image_size, split="val"),
    )
    test_ds = EmotionMultimodalDataset(
        split_samples["test"], modality=modality, tokenizer=tokenizer,
        image_transform=get_image_transforms(cfg.image.image_size, split="test"),
    )
    return DatasetBundle(train_ds, val_ds, test_ds, tokenizer, label_names)


def build_dataloaders(
    image_root: str,
    text_csv: str,
    modality: str = "multimodal",
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    dataset_type: str = "meld",
    meld_root: str = "dataset/meld",
    frame_root: Optional[str] = None,
):
    bundle = build_datasets(
        image_root=image_root,
        text_csv=text_csv,
        modality=modality,
        seed=seed,
        dataset_type=dataset_type,
        meld_root=meld_root,
        frame_root=frame_root,
    )

    train_loader = DataLoader(bundle.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(bundle.val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(bundle.test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    for loader in (train_loader, val_loader, test_loader):
        loader.label_names = bundle.label_names
        loader.tokenizer = bundle.tokenizer

    return train_loader, val_loader, test_loader
