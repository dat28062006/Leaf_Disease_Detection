"""
new_model/main.py - Mô hình nâng cấp (EfficientNet + DAAM + Swin Transformer)
Chạy train: python main.py --mode train
Chạy test : python main.py --mode test
"""

# ═══════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision.models.swin_transformer import SwinTransformerBlock
from timm.data import Mixup
import os
import pickle
import argparse
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 30
BASE_PATH  = ".."           # Đường dẫn tới thư mục chứa train.csv và ảnh
TRAIN_CSV  = f"{BASE_PATH}/train.csv"
TEST_CSV   = f"{BASE_PATH}/test.csv"
SAVE_DIR   = f"{BASE_PATH}/checkpoints_new"
CKPT_PATH  = f"{SAVE_DIR}/last.pth"
BEST_PATH  = f"{BASE_PATH}/best_new.pth"
LE_PATH    = f"{BASE_PATH}/label_encoder.pkl"

os.makedirs(SAVE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# KIẾN TRÚC MODEL (EfficientNet + DAAM + Swin Transformer)
# ═══════════════════════════════════════════════════════════════
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("efficientnet_b4", pretrained=True, features_only=True)

    def forward(self, x):
        return self.model(x)[-1]


class FocalLoss(nn.Module):
    """Trừng phạt nặng khi dự đoán sai các lớp bệnh hiếm."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=inputs.size(-1)).float()
        probs        = F.softmax(inputs, dim=-1)
        pt           = torch.sum(probs * targets, dim=-1)
        log_probs    = F.log_softmax(inputs, dim=-1)
        loss         = -torch.sum(targets * log_probs, dim=-1)
        focal_weight = (1 - pt) ** self.gamma
        return (self.alpha * focal_weight * loss).mean()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class DAAM(nn.Module):
    """Disease Attention Activation Module (Channel + Spatial)."""
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class SwinBlockWrapper(nn.Module):
    """2 tầng Swin Transformer với Shifted Window đúng chuẩn."""
    def __init__(self, dim, num_heads=8, window_size=[7, 7]):
        super().__init__()
        self.proj  = nn.Linear(dim, dim)
        shift      = [window_size[0] // 2, window_size[1] // 2]
        common_kw  = dict(dim=dim, num_heads=num_heads, window_size=window_size,
                          mlp_ratio=4.0, dropout=0.1, attention_dropout=0.1,
                          norm_layer=nn.LayerNorm)
        self.block1 = SwinTransformerBlock(**common_kw, shift_size=[0, 0])
        self.block2 = SwinTransformerBlock(**common_kw, shift_size=shift)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.proj(x)
        x = self.block1(x)
        x = self.block2(x)
        return x.mean(dim=[1, 2])   # (B, C)


class Model(nn.Module):
    """EfficientNet-B4 → DAAM → Swin Transformer → Classifier"""
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNBackbone()
        with torch.no_grad():
            out = self.cnn(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
            c, H, W = out.shape[1], out.shape[2], out.shape[3]
        self.daam       = DAAM(c)
        self.swin       = SwinBlockWrapper(dim=c, window_size=[H, W])
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.daam(x)
        x = self.swin(x)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════
# DATASET & TRANSFORMS
# ═══════════════════════════════════════════════════════════════
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
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(BASE_PATH, row["image"])
        img      = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, row["label"]


class TestDataset(Dataset):
    def __init__(self, df, le, transform=None):
        self.df        = df.reset_index(drop=True)
        self.le        = le
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(BASE_PATH, row["image"])
        img      = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        label = self.le.transform([row["plant_disease"]])[0]
        return img, label


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def make_perm(n, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(n, generator=g).tolist()


def save_ckpt(model, optimizer, scaler, scheduler, epoch, step, best_score, perm):
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
        "epoch": epoch, "step": step, "best_score": best_score, "perm": perm,
    }, CKPT_PATH)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast("cuda"):
                out = model(imgs)
            preds    = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ═══════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════
def run_train():
    print("Device:", DEVICE)
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

    train_dataset = CSVDataset(train_df, train_tf)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True,
                               prefetch_factor=4, drop_last=True)
    val_loader    = DataLoader(CSVDataset(val_df, val_tf), batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=4, pin_memory=True,
                               persistent_workers=True)

    model     = Model(num_classes).to(DEVICE)
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler    = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    mixup_fn  = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                      mode='batch', label_smoothing=0.1, num_classes=num_classes)

    start_epoch = global_step = 0
    best_score  = 0.0

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        best_score  = ckpt["best_score"]
        perm        = ckpt["perm"]
        print(f"Resumed epoch {start_epoch} | best={best_score:.4f}")
    else:
        perm = make_perm(len(train_dataset))

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, y in pbar:
            imgs, y = imgs.to(DEVICE), y.to(DEVICE)
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
            pbar.set_postfix({"loss": f"{float(loss):.4f}"})
            if global_step % 50 == 0:
                save_ckpt(model, optimizer, scaler, scheduler, epoch, global_step, best_score, perm)

        val_acc = evaluate(model, val_loader)
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} | val_acc={val_acc:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_score:
            best_score = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_acc": val_acc, "num_classes": num_classes}, BEST_PATH)
            print(f"  ✓ Saved best (acc={val_acc:.4f})")

        save_ckpt(model, optimizer, scaler, scheduler, epoch, global_step, best_score, perm)

    print(f"\nDone. Best val_acc = {best_score:.4f}")


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════
def run_test():
    print("Device:", DEVICE)
    if not os.path.exists(LE_PATH):
        raise FileNotFoundError(f"Không tìm thấy '{LE_PATH}'. Hãy chạy --mode train trước!")
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)

    if not os.path.exists(BEST_PATH):
        raise FileNotFoundError(f"Không tìm thấy '{BEST_PATH}'. Hãy chạy --mode train trước!")
    ckpt = torch.load(BEST_PATH, map_location=DEVICE)
    model = Model(num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model from epoch {ckpt.get('epoch','?')} | val_acc={ckpt.get('val_acc',0):.4f}")

    test_df = pd.read_csv(TEST_CSV)
    test_df["image"] = test_df["image"].str.replace("\\", "/", regex=False)
    missing = set(test_df["plant_disease"]) - set(le.classes_)
    if missing:
        print(f"⚠ Classes không thấy khi train: {missing}")

    test_loader = DataLoader(TestDataset(test_df, le, val_tf), batch_size=32,
                             shuffle=False, num_workers=4, pin_memory=True,
                             persistent_workers=True)

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast("cuda"):
                out = model(imgs)
            preds = out.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n" + "="*55)
    print("        KẾT QUẢ ĐÁNH GIÁ (NEW MODEL)")
    print("="*55)
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print("="*55)
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leaf Disease Detection - New Model")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    else:
        run_test()
