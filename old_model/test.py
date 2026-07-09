"""
old_model/test.py - Đánh giá mô hình gốc (EfficientNet-B4 + ViT)
Chạy: python test.py
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pickle
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

from model import Model

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE  = 224
BASE_PATH = ".."
TEST_CSV  = f"{BASE_PATH}/test.csv"
BEST_PATH = f"{BASE_PATH}/best_old.pth"
LE_PATH   = f"{BASE_PATH}/label_encoder_old.pkl"

print("Device:", DEVICE)

# ─────────────────────────────────────────────
# Load Label Encoder
# ─────────────────────────────────────────────
if not os.path.exists(LE_PATH):
    raise FileNotFoundError(f"Không tìm thấy '{LE_PATH}'. Hãy chạy train.py trước!")
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)
num_classes = len(le.classes_)
print(f"Loaded LabelEncoder: {num_classes} classes")

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
if not os.path.exists(BEST_PATH):
    raise FileNotFoundError(f"Không tìm thấy '{BEST_PATH}'. Hãy chạy train.py trước!")
ckpt  = torch.load(BEST_PATH, map_location=DEVICE)
model = Model(num_classes=num_classes).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded model from epoch {ckpt.get('epoch','?')} | val_acc={ckpt.get('val_acc', 0):.4f}")

# ─────────────────────────────────────────────
# Transform (giữ nguyên bản gốc)
# ─────────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(os.path.join(BASE_PATH, row["image"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = le.transform([row["plant_disease"]])[0]
        return img, label


test_df = pd.read_csv(TEST_CSV)
test_df["image"] = test_df["image"].str.replace("\\", "/", regex=False)

missing = set(test_df["plant_disease"]) - set(le.classes_)
if missing:
    print(f"⚠ Classes không thấy khi train: {missing}")

test_loader = DataLoader(
    TestDataset(test_df, val_tf), batch_size=32,
    shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
)

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds   = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("        KẾT QUẢ ĐÁNH GIÁ (OLD MODEL)")
print("="*55)
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print("="*55)
print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
