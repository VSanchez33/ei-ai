import time
import numpy as np ##
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

start_time = time.time()

print("---DATA STEP---")
data_dir = "./data/dataset" ## change this to your dataset path

# Transforms
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = full_dataset.classes
print("Classes found: ", class_names)

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

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)


## adding here to compute class counts and weights for weighted loss and oversampling to solve the issue of the model failing for some classes due to class imbalance
# counting samples per class in the training set
num_classes = len(class_names)
class_counts = [0] * num_classes
for idx in train_indices:
    label = full_dataset.targets[idx]
    class_counts[label] += 1
print("Training samples per class:", {class_names[i]: class_counts[i] for i in range(num_classes)})

# removing weighted random sampler because was too aggressive with the other method

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler) # now uses sampler to balance classes during training
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print("---MODEL CREATION STEP---")
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used: ", device)

model = EmotionCNN(num_classes=len(class_names)).to(device)

## modifying the nn.CrossEntropyLoss to the class-weighted version to give more importance to minority classes during training
class_weights = torch.tensor([1.0 / class_counts[i] for i in range(num_classes)], dtype=torch.float).to(device)
class_weights = class_weights / class_weights.sum() * num_classes  # normalize weights to sum to num_classes
criterion = nn.CrossEntropyLoss(weight=class_weights)
print("Class weights for loss:", {class_names[i]: class_weights[i].item() for i in range(num_classes)})

# removing weighted random sampler
class_counts_tensor = torch.tensor(class_counts, dtype=torch.float)
class_weights = 1.0 / torch.sqrt(class_counts_tensor)  # using sqrt to reduce the aggressiveness of the weights
class_weights = class_weights / class_weights.sum() * num_classes 
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
print("Class weights for loss:", {class_names[i]: class_weights[i].item() for i in range(num_classes)}) 

optimizer = optim.Adam(model.parameters(), lr=0.001)

## adding a learning rate scheduler to reduce the learning rate if the validation loss plateaus (like it did before after epoch 15), which can help the model converge better especially with imbalanced data
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # reduce LR by half every 5 epochs

print("---TRAINING STEP---")
for epoch in range(30):
    model.train()
    total_loss = 0
    for images, labels, in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    ## adding validation loss tracking to monitor overfitting and adjust learning rate if needed
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    scheduler.step()  # update learning rate based on scheduler

print("---TEST EVALUATION STEP---")
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

print("Test Accuracy:", sum(np.array(all_preds)==np.array(all_labels))/len(all_labels))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"Total training time: {hours}h {minutes}m {seconds}s")