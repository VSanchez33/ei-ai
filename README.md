# Quick Start Guide

## Initial Project Setup

### 1. Clone Repository

```bash
git clone https://github.com/NMSU-CSCI-4425-5425/project-ei-ai.git
cd src
```

---

## 2. Create Virtual Environment

### Linux / macOS

```bash
python -m venv eienv
source eienv/bin/activate
```

### Windows

```bash
python -m venv eienv
eienv\Scripts\activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Required External Dependency

## FFmpeg

FFmpeg is required for MELD video frame extraction.

### Ubuntu / Debian

```bash
sudo apt install ffmpeg
```

### Windows

Download:
https://ffmpeg.org/download.html

---

# Full Project Structure

```text
src/
│
├── data/
│   ├── dataset.py
│   └── preprocessing.py
│
├── models/
│   ├── cnn_image.py
│   ├── transformer_text.py
│   ├── multimodal_fusion.py
│   └── gru_text.py
│
├── evaluation/
│   └── metrics.py
│
├── utils/
│   ├── config.py
│   └── helpers.py
│
├── dataset/
│   └── meld/
│       ├── MELD.Raw.tar.gz
│       ├── train_sent_emo.csv
│       ├── dev_sent_emo.csv
│       ├── test_sent_emo.csv
│       ├── train_splits/
│       ├── dev_splits_complete/
│       ├── output_repeated_splits_test/
│       └── frames/
│
├── checkpoints/
│
├── results/
│
├── prepare_meld.py
├── train.py
├── inference.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset Setup

## Download MELD Dataset

Official dataset link:
https://affective-meld.github.io/

Place the files into:

```text
dataset/meld/
```

Expected files:

```text
dataset/meld/
├── MELD.Raw.tar.gz
├── train_sent_emo.csv
├── dev_sent_emo.csv
└── test_sent_emo.csv
```

---

# Step-by-Step Execution Pipeline

# Step 1 — Prepare MELD Frames

```bash
python prepare_meld.py --meld_root dataset/meld
```

This will:
- extract MELD videos
- process train/dev/test splits
- extract one representative frame per utterance
- save frames into:

```text
dataset/meld/frames/
```

---

# Step 2 — Train the Model

## Multimodal Training (Recommended)

```bash
python train.py \
  --dataset_type meld \
  --meld_root dataset/meld \
  --modality multimodal \
  --epochs 20 \
  --batch_size 16 \
  --lr 2e-5
```

---

## Text-only Training

```bash
python train.py \
  --dataset_type meld \
  --meld_root dataset/meld \
  --modality text \
  --epochs 5 \
  --batch_size 8 \
  --lr 2e-5
```

---

## Image-only Training

```bash
python train.py \
  --dataset_type meld \
  --meld_root dataset/meld \
  --modality image \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-4
```

---

# Step 3 — Run Inference

```bash
python inference.py \
  --modality multimodal \
  --checkpoint checkpoints/best_model_multimodal.pt \
  --tokenizer_path checkpoints/tokenizer_multimodal.json \
  --labels anger,disgust,fear,joy,neutral,sadness,surprise \
  --text "I can't believe this happened!" \
  --image dataset/meld/frames/test/dia100_utt4.jpg
```

---

# Output Locations

## Saved Models

```text
checkpoints/
```


---

# Recommended Hardware

| Component | Recommendation |
|---|---|
| GPU | NVIDIA GPU (16GB+ VRAM preferred) |
| RAM | 16GB+ |
| Storage | 50GB+ free space |
| CUDA | Recommended for training |

---
