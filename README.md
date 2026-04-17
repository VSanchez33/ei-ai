The original project approach (v1) was diverted due to dataset misalignment issues. To see the early-stage progress and the logic behind the first attempt, please switch to the branch: v1-misaligned-data. This current main branch contains the corrected implementation.

# Transformer-based Multimodal Emotion Recognition on MELD

This project now targets the **MELD** dataset and uses a **Transformer text encoder** plus a **CNN image encoder** for text, image, and multimodal emotion recognition.

## What changed
- Switched the text branch from GRU to **DistilBERT**.
- Switched the data pipeline from unpaired Kaggle image/text datasets to **MELD**, where text and video are aligned by utterance.
- Added `prepare_meld.py` to extract MELD videos and save one representative frame per utterance clip.

## Expected MELD layout
Place your files like this:

```text
project/
  dataset/
    meld/
      MELD.Raw.tar.gz
      train_sent_emo.csv
      dev_sent_emo.csv
      test_sent_emo.csv
```

## Step 1: prepare MELD
```bash
python prepare_meld.py --meld_root dataset/meld
```

This will:
- extract `MELD.Raw.tar.gz` if needed
- read videos from the standard MELD split folders
- save one frame per utterance into `dataset/meld/frames/`

## Step 2: train
### Multimodal
```bash
python train.py --dataset_type meld --meld_root dataset/meld --modality multimodal --epochs 10 --batch_size 8 --lr 2e-5
```

### Text-only
```bash
python train.py --dataset_type meld --meld_root dataset/meld --modality text --epochs 5 --batch_size 8 --lr 2e-5
```

### Image-only
```bash
python train.py --dataset_type meld --meld_root dataset/meld --modality image --epochs 10 --batch_size 16 --lr 1e-4
```

## Optional: freeze text backbone
```bash
python train.py --dataset_type meld --meld_root dataset/meld --modality multimodal --freeze_text
```

## Inference
```bash
python inference.py \
  --modality multimodal \
  --checkpoint checkpoints/best_model_multimodal.pt \
  --tokenizer_path checkpoints/tokenizer_multimodal.json \
  --labels anger,disgust,fear,joy,neutral,sadness,surprise \
  --text "I can't believe this happened!" \
  --image dataset/meld/frames/test/dia100_utt4.jpg
```

## Notes
- `ffmpeg` is required for `prepare_meld.py`.
- The first Hugging Face model load will download `distilbert-base-uncased`.
- MELD uses these 7 emotion labels: anger, disgust, fear, joy, neutral, sadness, surprise.
