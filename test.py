"""
test.py - Đánh giá mô hình nhận dạng bệnh lá cây trên tập test
Chạy: python test.py
"""
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from model import Model, IMG_SIZE

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "."
TEST_CSV  = f"{BASE_PATH}/test.csv"
BEST_PATH = f"{BASE_PATH}/best.pth"
LE_PATH   = f"{BASE_PATH}/label_encoder.pkl"

print("Device:", DEVICE)

# ─────────────────────────────────────────────
# Load Label Encoder
# ─────────────────────────────────────────────
if not os.path.exists(LE_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy '{LE_PATH}'. Hãy chạy train.py trước!"
    )
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)
num_classes = len(le.classes_)
print(f"Loaded LabelEncoder: {num_classes} classes")

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
if not os.path.exists(BEST_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy '{BEST_PATH}'. Hãy chạy train.py trước!"
    )

model = Model(num_classes=num_classes).to(DEVICE)
ckpt  = torch.load(BEST_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded best model from epoch {ckpt.get('epoch', '?')} | val_acc={ckpt.get('val_acc', '?'):.4f}")

# ─────────────────────────────────────────────
# Transform & Dataset
# ─────────────────────────────────────────────
test_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


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


test_df = pd.read_csv(TEST_CSV)
test_df["image"] = test_df["image"].str.replace("\\", "/", regex=False)

# Kiểm tra class chưa có trong LabelEncoder
missing = set(test_df["plant_disease"]) - set(le.classes_)
if missing:
    print(f"⚠ Các class trong test chưa thấy khi train: {missing}")

test_loader = DataLoader(
    TestDataset(test_df, test_tf),
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.amp.autocast("cuda"):
            outputs = model(imgs)

        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("           KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
print("="*50)
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print("="*50)
print("\nChi tiết từng lớp:")
print(classification_report(
    y_true, y_pred,
    target_names=le.classes_,
    zero_division=0
))
