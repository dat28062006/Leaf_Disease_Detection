"""
train.py - Huấn luyện mô hình nhận dạng bệnh lá cây
Chạy: python train.py
"""
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data import Mixup
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from model import Model, FocalLoss, IMG_SIZE

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
EPOCHS      = 30
BASE_PATH   = "."
TRAIN_CSV   = f"{BASE_PATH}/train.csv"
SAVE_DIR    = f"{BASE_PATH}/checkpoints"
CKPT_PATH   = f"{SAVE_DIR}/last.pth"
BEST_PATH   = f"{BASE_PATH}/best.pth"

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

import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

train_df, val_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df["label"]
)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Classes: {num_classes}")

# ─────────────────────────────────────────────
# Dataset & Transforms
# ─────────────────────────────────────────────
train_tf = A.Compose([
    A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomRain(p=0.2, drop_length=20, drop_width=1, blur_value=3),
    A.Spatter(p=0.2),
    A.RandomShadow(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                    min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


class CSVDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(BASE_PATH, row["image"])

        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, row["label"]


train_dataset = CSVDataset(train_df, train_tf)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True,
    prefetch_factor=4, drop_last=True
)
val_loader = DataLoader(
    CSVDataset(val_df, val_tf), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)

# ─────────────────────────────────────────────
# Model, Loss, Optimizer
# ─────────────────────────────────────────────
model     = Model(num_classes=num_classes).to(DEVICE)
criterion = FocalLoss(gamma=2.0, alpha=0.25)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scaler    = torch.amp.GradScaler("cuda")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
mixup_fn  = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
    mode='batch', label_smoothing=0.1, num_classes=num_classes
)

# ─────────────────────────────────────────────
# Checkpoint Helpers
# ─────────────────────────────────────────────
def make_perm(dataset_size, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(dataset_size, generator=g).tolist()


def save_ckpt(epoch, global_step, best_score, perm):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": global_step,
        "best_score": best_score,
        "perm": perm,
    }, CKPT_PATH)


# ─────────────────────────────────────────────
# Resume from checkpoint
# ─────────────────────────────────────────────
start_epoch = 0
global_step = 0
best_score  = 0.0

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["step"]
    best_score  = ckpt["best_score"]
    perm = ckpt["perm"]
    print(f"Resumed from epoch {start_epoch} | best={best_score:.4f}")
else:
    perm = make_perm(len(train_dataset))

# ─────────────────────────────────────────────
# Evaluate helper
# ─────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for imgs, labels in pbar:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
            preds    = outputs.argmax(dim=1)
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

    for batch_idx, (imgs, y) in enumerate(pbar):
        imgs = imgs.to(DEVICE)
        y    = y.to(DEVICE)

        imgs, y = mixup_fn(imgs, y)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out  = model(imgs)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        pbar.set_postfix({"loss": f"{float(loss):.4f}", "step": global_step})

        if global_step % 50 == 0:
            save_ckpt(epoch, global_step, best_score, perm)

    val_acc = evaluate(model, val_loader)
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | val_acc={val_acc:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    if val_acc > best_score:
        best_score = val_acc
        torch.save({
            "model":   model.state_dict(),
            "epoch":   epoch,
            "val_acc": val_acc,
            "num_classes": num_classes,
        }, BEST_PATH)
        print(f"  ✓ Best model saved (acc={val_acc:.4f})")

    save_ckpt(epoch, global_step, best_score, perm)

print(f"\nTraining complete. Best val_acc = {best_score:.4f}")
