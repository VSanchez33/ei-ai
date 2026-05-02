"""Inference for transformer-based text, image, or multimodal emotion prediction."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing import HFTextTokenizer, get_image_transforms
from models.multimodal_fusion import build_model
from utils.config import cfg
from utils.helpers import get_device, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Emotion Inference")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--modality", type=str, default="multimodal", choices=["text", "image", "multimodal"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer_multimodal.json")
    parser.add_argument("--labels", type=str, default="anger,disgust,fear,joy,neutral,sadness,surprise")
    return parser.parse_args()


def load_image(image_path: str) -> torch.Tensor:
    tf = get_image_transforms(cfg.image.image_size, split="test")
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0)


def main():
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    device = get_device()

    model = build_model(args.modality, num_classes=len(labels)).to(device)
    load_checkpoint(model, args.checkpoint, device=device)

    kwargs = {}
    if args.modality in {"text", "multimodal"}:
        if not args.text:
            raise ValueError("--text is required for text or multimodal inference")
        tokenizer = HFTextTokenizer.load(args.tokenizer_path)
        enc = tokenizer.encode_one(args.text)
        kwargs["text"] = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
        kwargs["attention_mask"] = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=device)

    if args.modality in {"image", "multimodal"}:
        if not args.image:
            raise ValueError("--image is required for image or multimodal inference")
        kwargs["image"] = load_image(args.image).to(device)

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(**kwargs), dim=-1).squeeze(0).cpu()

    top_idx = int(probs.argmax())
    print("Predicted emotion:", labels[top_idx])
    print("Confidence:", f"{float(probs[top_idx]):.2%}")
    print("All probabilities:")
    for label, prob in sorted(zip(labels, probs.tolist()), key=lambda x: -x[1]):
        print(f"  {label:<12} {prob:.4f}")


if __name__ == "__main__":
    main()
