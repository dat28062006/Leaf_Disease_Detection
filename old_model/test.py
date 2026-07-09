"""
old_model/test.py - Đánh giá mô hình gốc (EfficientNet-B4 + ViT)
Chạy: python test.py
Đánh giá trên tập test sử dụng Ensemble từ 5 Folds.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pickle
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

from model import Model
from train import PaperPreprocessing

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE  = 224
BASE_PATH = ".."
TEST_CSV  = f"{BASE_PATH}/test.csv"
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
# Transform (Tiền xử lý theo bài báo)
# ─────────────────────────────────────────────
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    PaperPreprocessing(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    TestDataset(test_df, test_tf), batch_size=32,
    shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
)

# ─────────────────────────────────────────────
# Inference (Ensemble 5 Folds)
# ─────────────────────────────────────────────
all_preds = np.zeros((len(test_df), num_classes))
y_true = []

# Chỉ trích xuất labels một lần
for _, labels in test_loader:
    y_true.extend(labels.numpy())

# Duyệt qua các model folds
folds_loaded = 0
for fold in range(1, 6):
    model_path = f"{BASE_PATH}/best_old_fold{fold}.pth"
    if not os.path.exists(model_path):
        print(f"Bỏ qua fold {fold} vì không tìm thấy checkpoint.")
        continue
    
    ckpt = torch.load(model_path, map_location=DEVICE)
    model = Model(num_classes=num_classes, freeze_layers=20).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    print(f"Đang infer với Fold {fold} (val_acc = {ckpt.get('val_acc', 0):.4f})...")
    
    fold_preds = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc=f"Testing Fold {fold}", leave=False):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                # Dùng softmax để cộng xác suất an toàn hơn cộng raw logits
                probs = torch.softmax(outputs, dim=1)
            fold_preds.extend(probs.cpu().numpy())
            
    all_preds += np.array(fold_preds)
    folds_loaded += 1

if folds_loaded == 0:
    raise FileNotFoundError("Không tìm thấy bất kỳ checkpoint fold nào. Hãy chạy train.py!")

# Tính trung bình xác suất
all_preds /= folds_loaded
y_pred = np.argmax(all_preds, axis=1)

# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(f"        KẾT QUẢ ĐÁNH GIÁ (OLD MODEL - ENSEMBLE {folds_loaded} FOLDS)")
print("="*60)
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print("="*60)
print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
