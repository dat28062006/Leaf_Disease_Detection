"""
new_model/test.py - Đánh giá mô hình nâng cấp (EfficientNet + DAAM + Swin Transformer)
Chạy: python test.py
"""
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

from model import Model, IMG_SIZE

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = ".."
TEST_CSV  = f"{BASE_PATH}/test.csv"
BEST_PATH = f"{BASE_PATH}/best_new.pth"
LE_PATH   = f"{BASE_PATH}/label_encoder_new.pkl"

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
# Transform (ImageNet stats, không augment khi test)
# ─────────────────────────────────────────────
test_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
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
        row      = self.df.iloc[idx]
        img_path = os.path.join(BASE_PATH, row["image"])
        img      = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        label = le.transform([row["plant_disease"]])[0]
        return img, label


test_df = pd.read_csv(TEST_CSV)
test_df["image"] = test_df["image"].str.replace("\\", "/", regex=False)

missing = set(test_df["plant_disease"]) - set(le.classes_)
if missing:
    print(f"⚠  Classes không thấy khi train: {missing}")

test_loader = DataLoader(
    TestDataset(test_df, test_tf), batch_size=32,
    shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
)

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast("cuda"):
            out = model(imgs)
        preds = out.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("        KẾT QUẢ ĐÁNH GIÁ (NEW MODEL)")
print("="*55)
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print("="*55)
print("\nChi tiết từng lớp bệnh:")
print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
