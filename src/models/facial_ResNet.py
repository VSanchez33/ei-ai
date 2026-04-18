import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

start_time = time.time()

data_dir = "./data/dataset"

print("Using Transfer Learning with ResNet-18 pretrained on ImageNet")
# Transforms — ResNet expects 224x224 input and ImageNet normalization
# (different from the scratch CNN which used 48x48 and 0.5 normalization)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet mean
                         std=[0.229, 0.224, 0.225]) # ImageNet std
])

transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load full dataset with train transforms first to get targets/indices
full_dataset = datasets.ImageFolder(data_dir, transform=transform_train)
class_names = full_dataset.classes
num_classes = len(class_names)
print("Classes found:", class_names)

# Train/Val/Test Split (70/15/15)
indices = list(range(len(full_dataset)))
targets = full_dataset.targets

train_indices, temp_indices = train_test_split(
    indices, test_size=0.3, stratify=targets, random_state=42
)
temp_targets = [targets[i] for i in temp_indices]
val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, stratify=temp_targets, random_state=42
)

# Val and test sets use a separate dataset object with val/test transforms
# (no augmentation bc we only augment training data)
full_dataset_eval = datasets.ImageFolder(data_dir, transform=transform_val_test)

train_dataset = Subset(full_dataset,      train_indices)
val_dataset   = Subset(full_dataset_eval, val_indices)
test_dataset  = Subset(full_dataset_eval, test_indices)

# Count samples per class in training set
class_counts = [0] * num_classes
for idx in train_indices:
    label = targets[idx]
    class_counts[label] += 1
print("Training samples per class:", {class_names[i]: class_counts[i] for i in range(num_classes)})

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)

print("MODEL CREATION STEP: Loading pretrained ResNet-18 and modifying final layer for 6 classes")

# Load pretrained ResNet-18 with ImageNet weights
# All the early conv layers already know how to detect edges, textures, shapes. Keep those frozen and only train the final classification head
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer with one sized for our 6 classes
# This is the only layer that will be trained in phase 1
in_features = model.fc.in_features # 512 for ResNet-18
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used:", device)
model = model.to(device)

# Weighted loss using sqrt of inverse frequency (same as facial_recognition.py)
class_counts_tensor = torch.tensor(class_counts, dtype=torch.float)
class_weights = 1.0 / torch.sqrt(class_counts_tensor)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
print("Class weights:", {class_names[i]: f"{class_weights[i].item():.4f}" for i in range(num_classes)})

# Phase 1: only train the new classification head (frozen backbone)
# Use a higher lr since we're training from scratch on a small layer
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# Phase 1 of training (head only, 10 epochs)
print("\nTRAINING PHASE 1: Classification Head Only (10 epochs)")
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            val_loss += criterion(model(images), labels).item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    scheduler.step()

# Phase 2 of training (unfreeze and fine-tune, 20 epochs)
# Now unfreeze the last ResNet block (layer4) and train end-to-end with a much smaller learning rate so we don't destroy the pretrained weights
print("\n---TRAINING PHASE 2: Fine-tuning ResNet layer4 + head (20 epochs)---")

for param in model.layer4.parameters(): # unfreeze only the last conv block
    param.requires_grad = True

# Lower lr for the backbone, higher for the new head
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3}
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_val_loss = float('inf')
best_model_state = None

for epoch in range(20):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            val_loss += criterion(model(images), labels).item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    scheduler.step()

    # Save best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"          -> New best model saved (val loss: {best_val_loss:.4f})")

# Load best model before evaluating on test set
print("\nLoading best model for test evaluation...")
model.load_state_dict(best_model_state)

print("TEST EVALUATION")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

print("Test Accuracy:", sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

total_time = time.time() - start_time
hours   = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"Total training time: {hours}h {minutes}m {seconds}s")