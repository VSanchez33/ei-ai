import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score

start_time = time.time()

#Load data
print("---DATA STEP---")
data = pd.read_csv('data/emotions.csv')
X = data['text'].tolist()
y = data['label'].tolist()

#Training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

#Tokenization and encoding
print("---TOKENIZATION AND ENCODING STEP---")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def encode(texts):
    return tokenizer(
        texts,
        padding = 'max_length',
        truncation = True,
        max_length = 128,
        return_tensors = 'pt'
    )

train_encodings = encode(X_train)
test_encodings = encode(X_test)

#Tensors
print("---TENSOR CREATION STEP---")
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#Model
print("---MODEL CREATION STEP---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device used: ", device)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
model.to(device)

#Optimizer and loss
print("---OPTIMIZER AND LOSS FUNCTION STEP---")
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

#Training loop
print("---TRAINING STEP---")
for epoch in range(10):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

#Evaluation
model.eval()
all_preds = []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())

acc = accuracy_score(y_test, all_preds)
print(f"Test Accuracy: {acc:.4f}")

total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"Total training time: {hours}h {minutes}m {seconds}s")