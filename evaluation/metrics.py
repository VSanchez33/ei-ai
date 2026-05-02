"""
evaluation/metrics.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive evaluation suite for the emotion detection system.

Metrics computed:
  • Accuracy (overall + per-class)
  • Loss (cross-entropy)
  • Precision, Recall, F1 (macro + weighted)
  • Matthews Correlation Coefficient (MCC)
  • Cohen's Kappa
  • Confusion Matrix (normalized + raw)
  • ROC AUC (one-vs-rest, macro)
  • Top-2 Accuracy

Charts generated (saved as PNG):
  1. Training history — loss + accuracy curves
  2. Confusion matrix heatmap (normalized)
  3. Per-class F1 bar chart
  4. ROC curves (one per class + macro average)
  5. Precision-Recall curves
  6. Prediction confidence distribution
  7. Combined evaluation dashboard
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, matthews_corrcoef,
    cohen_kappa_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from tabulate import tabulate

from utils.config import EMOTIONS, NUM_CLASSES


# ── Palette ────────────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#4361EE",
    "secondary": "#F72585",
    "accent":    "#7209B7",
    "success":   "#06D6A0",
    "warning":   "#FFB703",
    "neutral":   "#ADB5BD",
    "bg":        "#0F0F1A",
    "surface":   "#1A1A2E",
    "text":      "#E9ECEF",
}

EMOTION_COLORS = [
    "#E63946", "#FF6B35", "#F7D002", "#06D6A0",
    "#4895EF", "#7209B7", "#F72585",
]


def _set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    "#2D2D4E",
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["neutral"],
        "ytick.color":       PALETTE["neutral"],
        "text.color":        PALETTE["text"],
        "grid.color":        "#2D2D4E",
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "legend.facecolor":  PALETTE["surface"],
        "legend.edgecolor":  "#2D2D4E",
    })


# ── Inference Helper ───────────────────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over a DataLoader.
    Returns: (y_true, y_pred, y_prob) — all numpy arrays.
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for batch in loader:
        labels = batch["label"].to(device)

        # Build forward kwargs dynamically (works for all model variants)
        kwargs = {}
        if "image"        in batch: kwargs["image"]        = batch["image"].to(device)
        if "text"         in batch: kwargs["text"]         = batch["text"].to(device)
        if "attention_mask" in batch: kwargs["attention_mask"] = batch["attention_mask"].to(device)

        logits = model(**kwargs)                    # (B, C)
        probs  = torch.softmax(logits, dim=-1)      # (B, C)
        preds  = probs.argmax(dim=-1)               # (B,)

        all_true.extend(labels.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_prob.extend(probs.cpu().numpy())

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_prob),
    )


# ── Core Metrics ───────────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str] = EMOTIONS,
) -> Dict:
    classes = list(range(len(labels)))

    # Binarize for ROC AUC
    y_bin = label_binarize(y_true, classes=classes)

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall    = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1        = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_support   = np.bincount(y_true, minlength=len(labels))

    # ROC AUC (macro one-vs-rest)
    try:
        roc_auc_macro = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
        roc_auc_per   = roc_auc_score(y_bin, y_prob, average=None, multi_class="ovr")
    except Exception:
        roc_auc_macro = float("nan")
        roc_auc_per   = [float("nan")] * len(labels)

    # Top-2 accuracy
    top2_acc = np.mean(
        [y_true[i] in np.argsort(y_prob[i])[-2:] for i in range(len(y_true))]
    )

    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "top2_accuracy":     top2_acc,
        "precision_macro":   precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall_macro":      recall_score(y_true, y_pred,    average="macro",    zero_division=0),
        "f1_macro":          f1_score(y_true, y_pred,        average="macro",    zero_division=0),
        "f1_weighted":       f1_score(y_true, y_pred,        average="weighted", zero_division=0),
        "mcc":               matthews_corrcoef(y_true, y_pred),
        "kappa":             cohen_kappa_score(y_true, y_pred),
        "roc_auc_macro":     roc_auc_macro,
        # Per-class
        "per_class_precision": per_class_precision,
        "per_class_recall":    per_class_recall,
        "per_class_f1":        per_class_f1,
        "per_class_support":   per_class_support,
        "per_class_roc_auc":   np.array(roc_auc_per),
        "confusion_matrix":    confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ── Console Report ─────────────────────────────────────────────────────────────
def print_metrics_table(metrics: Dict, labels: List[str] = EMOTIONS):
    print("\n" + "═" * 60)
    print("  EMOTION DETECTION — EVALUATION REPORT")
    print("═" * 60)

    # Overall summary
    summary = [
        ["Accuracy",              f"{metrics['accuracy']:.4f}"],
        ["Top-2 Accuracy",        f"{metrics['top2_accuracy']:.4f}"],
        ["Precision (macro)",     f"{metrics['precision_macro']:.4f}"],
        ["Recall (macro)",        f"{metrics['recall_macro']:.4f}"],
        ["F1 (macro)",            f"{metrics['f1_macro']:.4f}"],
        ["F1 (weighted)",         f"{metrics['f1_weighted']:.4f}"],
        ["MCC",                   f"{metrics['mcc']:.4f}"],
        ["Cohen's Kappa",         f"{metrics['kappa']:.4f}"],
        ["ROC AUC (macro OvR)",   f"{metrics['roc_auc_macro']:.4f}"],
    ]
    print(tabulate(summary, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    # Per-class breakdown
    print("\n  PER-CLASS BREAKDOWN")
    per_class = []
    for i, label in enumerate(labels):
        per_class.append([
            label,
            f"{metrics['per_class_precision'][i]:.4f}",
            f"{metrics['per_class_recall'][i]:.4f}",
            f"{metrics['per_class_f1'][i]:.4f}",
            f"{metrics['per_class_roc_auc'][i]:.4f}" if not np.isnan(metrics['per_class_roc_auc'][i]) else "N/A",
            int(metrics['per_class_support'][i]),
        ])
    print(tabulate(
        per_class,
        headers=["Emotion", "Precision", "Recall", "F1", "ROC AUC", "Support"],
        tablefmt="fancy_grid",
    ))
    print()


# ── Chart 1: Training History ──────────────────────────────────────────────────
def plot_training_history(history: Dict, save_path: str):
    _set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold", color=PALETTE["text"], y=1.02)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=PALETTE["primary"],   linewidth=2, label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"],   color=PALETTE["secondary"], linewidth=2, label="Val Loss",   marker="s", markersize=3)
    ax.set_title("Loss Curve");  ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], color=PALETTE["success"],  linewidth=2, label="Train Acc", marker="o", markersize=3)
    ax.plot(epochs, history["val_acc"],   color=PALETTE["warning"],  linewidth=2, label="Val Acc",   marker="s", markersize=3)
    ax.set_title("Accuracy Curve"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


# ── Chart 2: Confusion Matrix ──────────────────────────────────────────────────
def plot_confusion_matrix(metrics: Dict, save_path: str, labels: List[str] = EMOTIONS):
    _set_dark_style()
    cm = metrics["confusion_matrix"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold", color=PALETTE["text"])

    for ax, data, title, fmt in zip(
        axes, [cm, cm_norm], ["Raw Counts", "Normalized (Row %)"], ["d", ".2f"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, xticklabels=labels, yticklabels=labels,
            cmap="magma", ax=ax, linewidths=0.5, linecolor="#2D2D4E",
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, color=PALETTE["text"])
        ax.set_xlabel("Predicted", color=PALETTE["text"])
        ax.set_ylabel("True", color=PALETTE["text"])
        ax.tick_params(colors=PALETTE["neutral"])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


# ── Chart 3: Per-Class F1 Bar Chart ───────────────────────────────────────────
def plot_per_class_metrics(metrics: Dict, save_path: str, labels: List[str] = EMOTIONS):
    _set_dark_style()
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, metrics["per_class_precision"], width, label="Precision", color=PALETTE["primary"],   alpha=0.9)
    ax.bar(x,          metrics["per_class_recall"],    width, label="Recall",    color=PALETTE["success"],   alpha=0.9)
    ax.bar(x + width,  metrics["per_class_f1"],        width, label="F1 Score",  color=PALETTE["secondary"], alpha=0.9)

    ax.set_title("Per-Class Precision / Recall / F1", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.legend(); ax.grid(True, axis="y")

    # Annotate F1 values
    for i, v in enumerate(metrics["per_class_f1"]):
        ax.text(i + width, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8, color=PALETTE["text"])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


# ── Chart 4: ROC Curves ────────────────────────────────────────────────────────
def plot_roc_curves(metrics: Dict, save_path: str, labels: List[str] = EMOTIONS):
    _set_dark_style()
    y_true = metrics["y_true"]
    y_prob = metrics["y_prob"]
    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.50)")

    macro_tpr, macro_fpr = np.zeros(300), np.linspace(0, 1, 300)
    for i, (label, color) in enumerate(zip(labels, EMOTION_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = metrics["per_class_roc_auc"][i]
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC={auc:.3f})")
        macro_tpr += np.interp(macro_fpr, fpr, tpr)

    macro_tpr /= n_classes
    macro_auc = metrics["roc_auc_macro"]
    ax.plot(macro_fpr, macro_tpr, color="white", linewidth=3, linestyle="--",
            label=f"Macro Average (AUC={macro_auc:.3f})")

    ax.set_title("ROC Curves — One vs. Rest", fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9); ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


# ── Chart 5: Precision-Recall Curves ──────────────────────────────────────────
def plot_pr_curves(metrics: Dict, save_path: str, labels: List[str] = EMOTIONS):
    _set_dark_style()
    y_true = metrics["y_true"]
    y_prob = metrics["y_prob"]
    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (label, color) in enumerate(zip(labels, EMOTION_COLORS)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{label} (AP={ap:.3f})")

    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9); ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


# ── Chart 6: Confidence Distribution ──────────────────────────────────────────
def plot_confidence_distribution(metrics: Dict, save_path: str, labels: List[str] = EMOTIONS):
    _set_dark_style()
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    y_prob = metrics["y_prob"]

    max_conf = y_prob.max(axis=1)
    correct = (y_true == y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confidence histogram: correct vs incorrect
    ax = axes[0]
    ax.hist(max_conf[correct],  bins=30, color=PALETTE["success"],   alpha=0.7, label="Correct", density=True)
    ax.hist(max_conf[~correct], bins=30, color=PALETTE["secondary"],  alpha=0.7, label="Incorrect", density=True)
    ax.set_title("Confidence Distribution: Correct vs. Incorrect")
    ax.set_xlabel("Max Softmax Confidence"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(True)

    # Mean confidence per class
    ax = axes[1]
    class_conf = [max_conf[y_true == c].mean() if (y_true == c).sum() > 0 else 0
                  for c in range(len(labels))]
    bars = ax.bar(labels, class_conf, color=EMOTION_COLORS[:len(labels)], alpha=0.9)
    ax.set_title("Mean Prediction Confidence per Class")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Mean Confidence"); ax.grid(True, axis="y")
    for bar, v in zip(bars, class_conf):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, color=PALETTE["text"])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


# ── Chart 7: Evaluation Dashboard ─────────────────────────────────────────────
def plot_dashboard(metrics: Dict, history: Dict, save_path: str, labels: List[str] = EMOTIONS):
    """
    Single combined figure: loss + acc curves, confusion matrix, F1 bars, overall metrics.
    """
    _set_dark_style()
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Multimodal Emotion Detection — Evaluation Dashboard",
                 fontsize=16, fontweight="bold", color=PALETTE["text"], y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: Loss + Accuracy ─────────────────────────────────────────────────
    epochs = range(1, len(history["train_loss"]) + 1)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(epochs, history["train_loss"], color=PALETTE["primary"],   lw=2, label="Train")
    ax0.plot(epochs, history["val_loss"],   color=PALETTE["secondary"], lw=2, label="Val")
    ax0.set_title("Loss"); ax0.set_xlabel("Epoch"); ax0.legend(); ax0.grid(True)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(epochs, history["train_acc"], color=PALETTE["success"],  lw=2, label="Train")
    ax1.plot(epochs, history["val_acc"],   color=PALETTE["warning"],  lw=2, label="Val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
    ax1.set_ylim(0, 1.05); ax1.legend(); ax1.grid(True)

    # ── Overall Metrics Radar-style bar (Row 0, Col 2) ────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    metric_names  = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "MCC\n(↕-1..1)", "Kappa"]
    metric_values = [
        metrics["accuracy"], metrics["precision_macro"], metrics["recall_macro"],
        metrics["f1_macro"], (metrics["mcc"] + 1) / 2,  # scale MCC to 0-1
        metrics["kappa"],
    ]
    colors = [PALETTE["primary"], PALETTE["success"], PALETTE["warning"],
              PALETTE["secondary"], PALETTE["accent"], PALETTE["neutral"]]
    bars = ax2.barh(metric_names, metric_values, color=colors, alpha=0.9)
    ax2.set_xlim(0, 1.1); ax2.set_title("Overall Metrics"); ax2.grid(True, axis="x")
    for bar, v in zip(bars, metric_values):
        ax2.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=9, color=PALETTE["text"])

    # ── Row 1: Confusion Matrix ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    cm_norm = metrics["confusion_matrix"].astype(float)
    cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels,
                cmap="magma", ax=ax3, linewidths=0.3, cbar_kws={"shrink": 0.7})
    ax3.set_title("Normalized Confusion Matrix"); ax3.set_xlabel("Predicted"); ax3.set_ylabel("True")

    # ── Per-class F1 (Row 1, Col 2) ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.barh(labels, metrics["per_class_f1"], color=EMOTION_COLORS, alpha=0.9)
    ax4.set_xlim(0, 1.1); ax4.set_title("Per-Class F1 Score"); ax4.grid(True, axis="x")
    for i, v in enumerate(metrics["per_class_f1"]):
        ax4.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9, color=PALETTE["text"])

    # ── Row 2: ROC + Confidence ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    y_bin = label_binarize(metrics["y_true"], classes=list(range(len(labels))))
    ax5.plot([0, 1], [0, 1], "k--", alpha=0.4)
    for i, (lbl, color) in enumerate(zip(labels, EMOTION_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], metrics["y_prob"][:, i])
        ax5.plot(fpr, tpr, color=color, lw=1.5, label=f"{lbl} ({metrics['per_class_roc_auc'][i]:.2f})")
    ax5.set_title(f"ROC Curves (Macro AUC={metrics['roc_auc_macro']:.3f})")
    ax5.set_xlabel("FPR"); ax5.set_ylabel("TPR")
    ax5.legend(fontsize=7, loc="lower right"); ax5.grid(True)

    ax6 = fig.add_subplot(gs[2, 2])
    correct = (metrics["y_true"] == metrics["y_pred"])
    max_conf = metrics["y_prob"].max(axis=1)
    ax6.hist(max_conf[correct],  bins=20, color=PALETTE["success"],  alpha=0.7, label="Correct",  density=True)
    ax6.hist(max_conf[~correct], bins=20, color=PALETTE["secondary"], alpha=0.7, label="Incorrect", density=True)
    ax6.set_title("Confidence Distribution"); ax6.set_xlabel("Max Probability")
    ax6.legend(); ax6.grid(True)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Dashboard saved → {save_path}")


# ── Master Evaluation Function ─────────────────────────────────────────────────
def full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    history: Dict,
    output_dir: str = "results",
    labels: List[str] = EMOTIONS,
) -> Dict:
    """
    Run all evaluations: collect predictions, compute metrics, print tables, save charts.
    """
    print("\n🔍 Running evaluation on test set...")
    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred, y_prob, labels)
    print_metrics_table(metrics, labels)

    # ── Save all charts ────────────────────────────────────────────────────────
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    print("\n📊 Generating charts...")

    plot_training_history(history,        str(od / "01_training_history.png"))
    plot_confusion_matrix(metrics,        str(od / "02_confusion_matrix.png"), labels)
    plot_per_class_metrics(metrics,       str(od / "03_per_class_metrics.png"), labels)
    plot_roc_curves(metrics,              str(od / "04_roc_curves.png"), labels)
    plot_pr_curves(metrics,              str(od / "05_pr_curves.png"), labels)
    plot_confidence_distribution(metrics, str(od / "06_confidence_distribution.png"), labels)
    plot_dashboard(metrics, history,      str(od / "07_evaluation_dashboard.png"), labels)

    print(f"\n✅ All results saved to: {output_dir}/\n")
    return metrics
