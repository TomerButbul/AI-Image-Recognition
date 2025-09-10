import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score

# =========================================================
# Config
# =========================================================
SEED = 42
DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
LR = 1e-4
PATIENCE = 3
MAX_EPOCHS = 50

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================================================
# Transforms
# =========================================================
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================================================
# Dataset
# =========================================================
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(DATA_DIR, self.data.iloc[idx, 1])
        label = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, label  # <-- always return CPU tensors

# =========================================================
# Data Split
# =========================================================
df = pd.read_csv(TRAIN_CSV)
train_size = int(0.8 * len(df))
test_size = len(df) - train_size
train_subset, test_subset = random_split(
    df.sample(frac=1, random_state=SEED), [train_size, test_size]
)

train_dataset = ImageDataset(df.iloc[train_subset.indices], transform)
test_dataset = ImageDataset(df.iloc[test_subset.indices], transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

# =========================================================
# Model
# =========================================================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

# =========================================================
# Training with Early Stopping
# =========================================================
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                patience=PATIENCE, max_epochs=MAX_EPOCHS):
    print("Starting training with early stopping...")

    best_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            images, labels = images.to(device), labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch} Validation", leave=False):
                images, labels = images.to(device), labels.unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}")

        # ---- Scheduler Step ----
        scheduler.step(avg_val_loss)

        # ---- Early Stopping ----
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, "best_model.pth")
            epochs_no_improve = 0
            print("Validation improved, saving model.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Restored best model state.")

# =========================================================
# Evaluation
# =========================================================
def evaluate_model(model, loader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.unsqueeze(1).to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds.extend(np.round(probs))
            actuals.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(actuals, preds)
    print(f"Test Accuracy: {acc:.4f}")
    return preds, actuals

# =========================================================
# Save Wrong Predictions
# =========================================================
def save_wrong_predictions(model, loader, save_dir="incorrect_predictions"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Saving incorrect predictions"):
            images, labels = images.to(device), labels.unsqueeze(1).to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = np.round(preds)

            incorrect = np.where(preds != labels.cpu().numpy().flatten())[0]
            for i in incorrect:
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(
                    os.path.join(save_dir, f"wrong_{time.time()}_{i}.png")
                )

    print(f"Incorrect predictions saved in {save_dir}")

# =========================================================
# Run everything
# =========================================================
train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)
evaluate_model(model, test_loader)
save_wrong_predictions(model, test_loader)
