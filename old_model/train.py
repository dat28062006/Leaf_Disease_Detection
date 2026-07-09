"""
old_model/train.py - Huấn luyện mô hình gốc theo đúng bài báo (EfficientNet-B4 + ViT)
Chạy: python train.py

Các thông số theo bài báo:
- Tiền xử lý: Kích thước 224x224, Chuẩn hóa ImageNet, Gaussian Blur, Canny, K-means.
- Tăng cường dữ liệu (chỉ dùng cho training): Xoay ±30°, Lật ngang (p=0.5), Zoom 10-20%, 
  Đổi màu 20%, Gaussian Noise σ=0.05.
- Kích thước lô (Batch Size): 32
- Số vòng lặp (Epochs): 10
- Thuật toán tối ưu: Adam (lr=1e-4)
- LR Scheduler: giảm ×0.1 sau epoch thứ 6
- Xử lý mất cân bằng lớp: Stratified Augmentation (WeightedRandomSampler tới mức trung vị)
- 5-Fold Stratified Cross-Validation: Đánh giá mô hình chéo trên 5 tập phân tầng.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from model import Model

# ─────────────────────────────────────────────
# Config (theo bài báo)
# ─────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 10          # Theo bài báo: 10 epoch đủ do backbone đã pretrain
LR         = 1e-4
BASE_PATH  = ".."
TRAIN_CSV  = f"{BASE_PATH}/train.csv"
SAVE_DIR   = f"{BASE_PATH}/checkpoints_old"
LE_PATH    = f"{BASE_PATH}/label_encoder_old.pkl"

os.makedirs(SAVE_DIR, exist_ok=True)
print("Device:", DEVICE)

# ─────────────────────────────────────────────
# Tiền xử lý (Preprocessing) theo bài báo
# ─────────────────────────────────────────────
class PaperPreprocessing:
    """
    Tiền xử lý theo bài báo:
    - Gaussian Blur để khử nhiễu.
    - Canny Edge + K-means Clustering để phân vùng vết bệnh khỏi nền.
    (Để tối ưu tốc độ huấn luyện, K-means được áp dụng với K=2 trên ảnh đã thu nhỏ)
    """
    def __call__(self, img_pil):
        img_np = np.array(img_pil)
        
        # 1. Gaussian Blur
        img_blur = cv2.GaussianBlur(img_np, (5, 5), 0)
        
        # 2. K-means Clustering (mô phỏng tách nền)
        Z = img_blur.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2 # Phân cụm nền và lá
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        img_kmeans = res.reshape((img_np.shape))
        
        # 3. Canny Edge (Tính viền và kết hợp nhẹ vào ảnh K-means để làm rõ đường nét)
        gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Kết hợp ảnh (K-means + viền Canny)
        img_final = cv2.addWeighted(img_kmeans, 0.8, edges_3c, 0.2, 0)
        return Image.fromarray(img_final)


class AddGaussianNoise:
    """Thêm nhiễu Gaussian (σ=0.05 theo bài báo)."""
    def __init__(self, std=0.05):
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

# ─────────────────────────────────────────────
# Augmentation (theo bài báo)
# Xoay ±30°, lật ngang, zoom 10-20%, đổi màu 20%, Gaussian noise σ=0.05
# ─────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    PaperPreprocessing(),                                           # Tiền xử lý (Blur + Kmeans + Canny)
    transforms.RandomHorizontalFlip(p=0.5),                         # Lật ngang (p=0.5)
    transforms.RandomRotation(30),                                  # Xoay ±30°
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),       # Zoom 10-20%
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2       # Đổi màu 20%
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet chuẩn
    AddGaussianNoise(std=0.05),                                     # Nhiễu Gaussian σ=0.05
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    PaperPreprocessing(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

# ─────────────────────────────────────────────
# Evaluate Function
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
# Main Pipeline
# ─────────────────────────────────────────────
def main():
    df = pd.read_csv(TRAIN_CSV)
    df["image"] = df["image"].str.replace("\\", "/", regex=False)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["plant_disease"])
    num_classes = len(le.classes_)

    with open(LE_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"Total samples: {len(df)} | Classes: {num_classes}")

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print(f"\n{'='*40}")
        print(f"          FOLD {fold + 1}/5")
        print(f"{'='*40}")

        train_df = df.iloc[train_idx].copy()
        val_df   = df.iloc[val_idx].copy()

        # Xử lý mất cân bằng lớp qua WeightedRandomSampler
        # Oversample lớp thiểu số lên bằng median class size
        label_counts = train_df["label"].value_counts().sort_index()
        median_count = int(label_counts.median())
        class_weights = 1.0 / label_counts.values.astype(float)
        sample_weights = torch.tensor(
            [class_weights[label] for label in train_df["label"].values],
            dtype=torch.float
        )
        
        num_samples = num_classes * median_count
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True
        )

        train_dataset = CSVDataset(train_df, train_tf)
        train_loader  = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=4, pin_memory=True, persistent_workers=True,
            prefetch_factor=4, drop_last=True
        )
        val_loader = DataLoader(
            CSVDataset(val_df, val_tf), batch_size=BATCH_SIZE,
            shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
        )

        # Model, Loss, Optimizer (Adam)
        model = Model(num_classes=num_classes, freeze_layers=20).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scaler = torch.amp.GradScaler("cuda")

        # LR Scheduler: giảm ×0.1 sau epoch thứ 6
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

        best_score = 0.0
        best_model_path = f"{BASE_PATH}/best_old_fold{fold+1}.pth"

        for epoch in range(EPOCHS):
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
                
                pbar.set_postfix({
                    "loss": f"{float(loss):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
                })

            val_acc = evaluate(model, val_loader)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{EPOCHS} | val_acc={val_acc:.4f} | lr={current_lr:.2e}")

            if val_acc > best_score:
                best_score = val_acc
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "num_classes": num_classes,
                }, best_model_path)
                print(f"  ✓ Best model fold {fold+1} saved (acc={val_acc:.4f})")
        
        fold_results.append(best_score)
        print(f"Fold {fold+1} complete. Best val_acc = {best_score:.4f}")

    print("\n" + "="*40)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("="*40)
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1}: {acc:.4f}")
    
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    print(f"\nFinal: {mean_acc:.4f} ± {std_acc:.4f}")


if __name__ == "__main__":
    main()
