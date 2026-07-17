import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Cài đặt SAM2: pip install git+https://github.com/facebookresearch/segment-anything-2.git
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("❌ Thư viện SAM2 chưa được cài đặt. Vui lòng chạy lệnh:")
    print("pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    exit(1)

# --- CONFIG ---
BASE_PATH = os.path.abspath(os.path.dirname(__file__))
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")

# Tải model SAM2 Tiny (Nhỏ nhất, tốn ít VRAM, chạy siêu nhanh trên RTX 4060)
CHECKPOINT = os.path.join(BASE_PATH, "sam2_hiera_tiny.pt")
MODEL_CFG = "configs/sam2/sam2_hiera_t.yaml"

if not os.path.exists(CHECKPOINT):
    print("❌ Vui lòng tải weights SAM2 Tiny. Chạy lệnh sau trong Terminal:")
    print("curl -O https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Khởi động SAM2 trên thiết bị: {device}")

# Kích hoạt tính năng tiết kiệm VRAM và tối ưu CUDA
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)

def process_dataset(csv_path, output_dir_name="SAM2_Masks"):
    if not os.path.exists(csv_path):
        print(f"⚠️ Không tìm thấy file {csv_path}")
        return

    df = pd.read_csv(csv_path)
    output_base = os.path.join(BASE_PATH, output_dir_name)
    os.makedirs(output_base, exist_ok=True)

    print(f"\n🔄 Bắt đầu sinh Mask cho {csv_path}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_rel_path = row['image'] # vd: Train/Apple___Apple_scab/abc.jpg
        img_abs_path = os.path.join(BASE_PATH, img_rel_path).replace("\\", "/")

        if not os.path.exists(img_abs_path):
            continue

        img = cv2.imread(img_abs_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # 1. Đưa ảnh vào SAM2
        predictor.set_image(img_rgb)

        # 2. Kết hợp Bounding Box 90% và 5 Điểm (1 tâm + 4 góc) để ép SAM2 bao trọn toàn bộ lá
        input_box = np.array([[int(W * 0.05), int(H * 0.05), int(W * 0.95), int(H * 0.95)]])
        input_point = np.array([
            [W // 2, H // 2],
            [W // 4, H // 4],
            [3 * W // 4, H // 4],
            [W // 4, 3 * H // 4],
            [3 * W // 4, 3 * H // 4]
        ])
        input_label = np.array([1, 1, 1, 1, 1]) # Tất cả đều là foreground (lá)

        # 3. Dự đoán
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True, # Lấy nhiều mask để lọc cái tốt nhất
        )

        # 4. Lấy mask có độ tự tin cao nhất
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx] # (H, W) boolean mask

        # 5. Chuyển thành ảnh đen trắng (0 và 255)
        mask_img = (best_mask * 255).astype(np.uint8)

        # 6. Chuẩn hóa đường dẫn lưu file (gom chung Train/Test vào đúng thư mục)
        path_parts = img_rel_path.replace("\\", "/").split("/")
        img_name = path_parts[-1]
        class_name = path_parts[-2]

        split_name = "Train" if "train.csv" in csv_path.lower() else "Test"
        out_rel_path = f"{split_name}/{class_name}/{img_name}"

        # Đổi đuôi sang .png an toàn
        out_rel_path = os.path.splitext(out_rel_path)[0] + ".png"

        out_abs_path = os.path.join(output_base, out_rel_path)
        os.makedirs(os.path.dirname(out_abs_path), exist_ok=True)

        cv2.imwrite(out_abs_path, mask_img)

if __name__ == "__main__":
    process_dataset(TRAIN_CSV)
    process_dataset(TEST_CSV)
    print("\n✅ Đã sinh xong toàn bộ SAM2 Masks!")
    print("📁 Kết quả được lưu tại thư mục: SAM2_Masks/")
