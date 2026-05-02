"""Training entry point for the transformer-based MELD multimodal emotion system."""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import build_dataloaders
from evaluation.metrics import full_evaluation
from models.multimodal_fusion import build_model, count_parameters
from utils.config import cfg
from utils.helpers import (
    EarlyStopping,
    MetricsTracker,
    get_device,
    get_logger,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)

logger = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer-based MELD emotion trainer")
    parser.add_argument("--modality", type=str, default="multimodal", choices=["text", "image", "multimodal", "audio"])
    parser.add_argument("--dataset_type", type=str, default="meld", choices=["meld"])
    parser.add_argument("--meld_root", type=str, default="dataset/meld", help="Path to MELD root containing CSV files and videos or MELD.Raw.tar.gz")
    parser.add_argument("--frame_root", type=str, default=None, help="Path to extracted MELD frames. Default: <meld_root>/frames")
    parser.add_argument("--image_root", type=str, default="dataset/images")
    parser.add_argument("--text_csv", type=str, default="dataset/text/emotions.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--freeze_text", action="store_true", help="Freeze transformer backbone during training")
    return parser.parse_args()


def _build_kwargs(batch, device, modality):
    kwargs = {}

    if modality in ("image", "multimodal", "audio"):
        kwargs["image"] = batch["image"].to(device)

    if modality in ("text", "multimodal", "audio"):
        kwargs["text"] = batch["text"].to(device)
        kwargs["attention_mask"] = batch["attention_mask"].to(device)

    return kwargs


def train_epoch(model, loader, optimizer, criterion, device, modality):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(loader, desc="  Train", leave=False, ncols=90):
        labels = batch["label"].to(device)
        kwargs = _build_kwargs(batch, device, modality)

        optimizer.zero_grad(set_to_none=True)
        logits = model(**kwargs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, modality):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(loader, desc="  Eval ", leave=False, ncols=90):
        labels = batch["label"].to(device)
        kwargs = _build_kwargs(batch, device, modality)

        logits = model(**kwargs)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Make output dirs early so tokenizer/checkpoint saving cannot fail later
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cfg.train.modality = args.modality
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.train.dataset_type = args.dataset_type
    cfg.text.freeze_backbone = args.freeze_text

    logger.info(f"Modality   : {args.modality.upper()}")
    logger.info(f"Dataset    : {args.dataset_type.upper()}")
    logger.info(f"MELD root  : {args.meld_root}")
    logger.info(f"Frame root : {args.frame_root or (Path(args.meld_root) / 'frames')}")

    train_loader, val_loader, test_loader = build_dataloaders(
        image_root=args.image_root,
        text_csv=args.text_csv,
        modality=args.modality,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=cfg.train.num_workers,
        dataset_type=args.dataset_type,
        meld_root=args.meld_root,
        frame_root=args.frame_root,
    )

    label_names = list(train_loader.label_names)
    logger.info(f"Classes    : {label_names}")
    logger.info(
        f"Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    tokenizer = getattr(train_loader, "tokenizer", None)
    if tokenizer is not None:
        tokenizer_path = str(Path(args.checkpoint_dir) / f"tokenizer_{args.modality}.json")
        tokenizer.save(tokenizer_path)
        logger.info(f"Tokenizer saved → {tokenizer_path}")

    model = build_model(args.modality, num_classes=len(label_names)).to(device)
    logger.info(f"Model      : {model.__class__.__name__} ({count_parameters(model):,} trainable parameters)")

    train_labels = [s.label_idx for s in train_loader.dataset.samples]
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / max(counts[i], 1) for i in range(len(label_names))],
        dtype=torch.float32,
        device=device,
    )
    logger.info(f"Class weights: {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.train.weight_decay)

    if cfg.train.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif cfg.train.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        scheduler = None

    early_stop = EarlyStopping(patience=cfg.train.patience, mode="min")
    tracker = MetricsTracker()

    best_ckpt = str(Path(args.checkpoint_dir) / f"best_model_{args.modality}.pt")

    if args.eval_only:
        ckpt_path = args.checkpoint or best_ckpt
        load_checkpoint(model, ckpt_path, device=device)
        history = {"train_loss": [0], "val_loss": [0], "train_acc": [0], "val_acc": [0]}
        full_evaluation(model, test_loader, device, history, args.output_dir, labels=label_names)
        return

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, args.modality)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, args.modality)

        if scheduler is not None:
            scheduler.step()

        tracker.update(train_loss, val_loss, train_acc, val_acc)

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_loss": val_loss, "val_acc": val_acc},
                best_ckpt,
            )

        if not args.no_early_stop and early_stop(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    load_checkpoint(model, best_ckpt, device=device)
    full_evaluation(model, test_loader, device, tracker.history, args.output_dir, labels=label_names)
    tracker.save(f"{args.output_dir}/training_history.json")


if __name__ == "__main__":
    main()