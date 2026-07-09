"""
old_model/train.py - Huấn luyện mô hình gốc (EfficientNet-B4 + ViT)
Chạy: python train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pickle
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from model import Model

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 10
BASE_PATH  = ".."
TRAIN_CSV  = f"{BASE_PATH}/train.csv"
SAVE_DIR   = f"{BASE_PATH}/checkpoints_old"
CKPT_PATH  = f"{SAVE_DIR}/last.pth"
BEST_PATH  = f"{BASE_PATH}/best_old.pth"
LE_PATH    = f"{BASE_PATH}/label_encoder_old.pkl"

os.makedirs(SAVE_DIR, exist_ok=True)
print("Device:", DEVICE)

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV)
df["image"] = df["image"].str.replace("\\", "/", regex=False)

le = LabelEncoder()
df["label"] = le.fit_transform(df["plant_disease"])
num_classes = len(le.classes_)

with open(LE_PATH, "wb") as f:
    pickle.dump(le, f)

train_df, val_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df["label"]
)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Classes: {num_classes}")

# ─────────────────────────────────────────────
# Transforms (giữ nguyên bản gốc notebook)
# ─────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class CSVDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(BASE_PATH, row["image"])
        img      = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["label"]


train_dataset = CSVDataset(train_df, train_tf)
train_loader  = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True,
    prefetch_factor=4, drop_last=True
)
val_loader = DataLoader(
    CSVDataset(val_df, val_tf), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
)

# ─────────────────────────────────────────────
# Model, Loss, Optimizer (giữ nguyên bản gốc)
# ─────────────────────────────────────────────
model     = Model(num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scaler    = torch.amp.GradScaler("cuda")


# ─────────────────────────────────────────────
# Checkpoint Helpers
# ─────────────────────────────────────────────
def make_perm(n, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(n, generator=g).tolist()


def save_ckpt(epoch, step, best_score, perm):
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch,
        "step": step, "best_score": best_score, "perm": perm,
    }, CKPT_PATH)


# ─────────────────────────────────────────────
# Resume
# ─────────────────────────────────────────────
start_epoch = global_step = 0
best_score  = 0.0

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["step"]
    best_score  = ckpt["best_score"]
    perm        = ckpt["perm"]
    print(f"Resumed from epoch {start_epoch}")
else:
    perm = make_perm(len(train_dataset))


# ─────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast("cuda"):
                out = model(imgs)
            preds    = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            pbar.set_postfix({"acc": f"{correct/total:.4f}"})
    return correct / total


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
for epoch in range(start_epoch, EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for imgs, y in pbar:
        imgs, y = imgs.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out  = model(imgs)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        pbar.set_postfix({"loss": f"{float(loss):.4f}", "step": global_step})
        if global_step % 50 == 0:
            save_ckpt(epoch, global_step, best_score, perm)

    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | val_acc={val_acc:.4f}")

    if val_acc > best_score:
        best_score = val_acc
        torch.save({
            "model": model.state_dict(), "epoch": epoch,
            "val_acc": val_acc, "num_classes": num_classes,
        }, BEST_PATH)
        print(f"  ✓ Best model saved (acc={val_acc:.4f})")

    save_ckpt(epoch, global_step, best_score, perm)

print(f"\nTraining complete. Best val_acc = {best_score:.4f}")
