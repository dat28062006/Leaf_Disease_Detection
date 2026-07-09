import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from timm.data import Mixup
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description="Leaf Disease Detection")
parser.add_argument("--mode", default="train", choices=["train", "test"], help="Run mode: train or test")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

BASE_PATH = "."

TRAIN_CSV = f"{BASE_PATH}/train.csv"
TEST_CSV = f"{BASE_PATH}/test.csv"
SAVE_DIR = f"{BASE_PATH}/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
CKPT_PATH = f"{SAVE_DIR}/last.pth"

print("Device:", DEVICE)

df = pd.read_csv(TRAIN_CSV)


df["image"] = df["image"].str.replace("\\", "/", regex=False)

le = LabelEncoder()
df["label"] = le.fit_transform(df["plant_disease"])

num_classes = len(le.classes_)

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df["label"]
)

print(len(train_df), len(val_df))

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

train_tf = A.Compose([
    A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomRain(p=0.2, drop_length=20, drop_width=1, blur_value=3),
    A.Spatter(p=0.2),
    A.RandomShadow(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

train_dataset = CSVDataset(train_df, train_tf)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
      persistent_workers=True,
    prefetch_factor=4,

    drop_last=True
)

val_loader = DataLoader(
    CSVDataset(val_df, val_tf),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            features_only=True
        )

    def forward(self, x):
        return self.model(x)[-1]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=inputs.size(-1)).float()
        
        probs = F.softmax(inputs, dim=-1)
        pt = torch.sum(probs * targets, dim=-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        loss = -torch.sum(targets * log_probs, dim=-1)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * loss
        
        return loss.mean()


from torchvision.models.swin_transformer import SwinTransformerBlock

class SwinBlockWrapper(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=[7, 7]):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        shift = [window_size[0] // 2, window_size[1] // 2]

        self.block1 = SwinTransformerBlock(
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=[0, 0], mlp_ratio=4.0, dropout=0.1, attention_dropout=0.1, norm_layer=nn.LayerNorm
        )
        self.block2 = SwinTransformerBlock(
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=shift, mlp_ratio=4.0, dropout=0.1, attention_dropout=0.1, norm_layer=nn.LayerNorm
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.proj(x)
        
        x = self.block1(x)
        x = self.block2(x)
        
        x = x.mean(dim=[1, 2]) # (B, C)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class DAAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(DAAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBackbone()

        with torch.no_grad():
            dummy_out = self.cnn(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
            c = dummy_out.shape[1]
            H, W = dummy_out.shape[2], dummy_out.shape[3]

        self.daam = DAAM(c)
        self.swin = SwinBlockWrapper(dim=c, window_size=[H, W])
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.daam(x)
        x = self.swin(x)
        return self.classifier(x)

model = Model().to(DEVICE)

criterion = FocalLoss(gamma=2.0, alpha=0.25)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

mixup_fn = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
    mode='batch', label_smoothing=0.1, num_classes=num_classes
)

def make_perm(dataset_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(dataset_size, generator=g).tolist()

def save_ckpt(epoch, global_step, best_score, perm):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": global_step,
        "best_score": best_score,
        "perm": perm   # 🔥 QUAN TRỌNG
    }, CKPT_PATH)

start_epoch = 0
global_step = 0
best_score = 0

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt["epoch"]
    global_step = ckpt["step"]
    best_score = ckpt["best_score"]
    perm = ckpt["perm"]

    print("Resumed OK")
else:
    perm = make_perm(len(train_dataset), seed=42)

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

import torch
from tqdm import tqdm

def evaluate(model, val_loader):
    model.eval()

    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for imgs, labels in pbar:

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.amp.autocast("cuda"):
                outputs = model(imgs)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = correct / total

            pbar.set_postfix({
                "acc": f"{acc:.4f}",
                "correct": correct,
                "total": total
            })

    return correct / total


if args.mode == "train":
  for epoch in range(start_epoch, EPOCHS):

    model.train()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, (imgs, y) in enumerate(pbar):

        imgs = imgs.to(DEVICE)
        y = y.to(DEVICE)

        imgs, y = mixup_fn(imgs, y)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            out = model(imgs)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1

        pbar.set_postfix({
            "step": global_step,
            "loss": float(loss)
        })

        if global_step % 50 == 0:
            save_ckpt(epoch, global_step, best_score, perm)

    val_acc = evaluate(model, val_loader)

    scheduler.step()

    print(f"Epoch {epoch} | val_acc={val_acc:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    if val_acc > best_score:
        best_score = val_acc
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc
        }, "best.pth")

    save_ckpt(epoch, global_step, best_score, perm)

class TestDataset(Dataset):
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

        label = le.transform([row["plant_disease"]])[0]

        return img, label

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if args.mode == "test":
    test_df = pd.read_csv(TEST_CSV)

    test_loader = DataLoader(
        TestDataset(test_df, val_tf),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    best_ckpt = torch.load("best.pth", map_location=DEVICE)
    model.load_state_dict(best_ckpt["model"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall   :", recall_score(y_true, y_pred, average="weighted"))
    print("F1-score :", f1_score(y_true, y_pred, average="weighted"))

    missing = set(test_df["plant_disease"]) - set(le.classes_)
    print("Missing classes:", missing)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    print("Epoch      :", ckpt["epoch"])
    print("Global step:", ckpt["step"])
    print("Best score :", ckpt["best_score"])