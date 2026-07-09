"""
old_model/model.py - Kiến trúc gốc từ bài báo (EfficientNet-B4 + ViT)

Thiết kế theo bài báo:
- EfficientNet-B4 pretrained làm Backbone trích xuất đặc trưng.
- Đóng băng (freeze) 20 lớp đầu, chỉ fine-tune các lớp còn lại.
- Cơ chế ghép nối (Fusion): Feature map -> Flatten -> Dropout(0.3) -> Linear -> ViT
- ViT Block KHÔNG có Positional Encoding (chủ ý của tác giả):
    Lý do: Đầu vào của ViT không phải ảnh thô mà là feature map từ CNN.
    Feature map từ EfficientNet đã mang sẵn cả thông tin không gian (spatial)
    và ngữ nghĩa (semantic), nên Transformer không cần PE thêm.
- Multi-head Self-Attention của ViT tập trung vào vùng bị bệnh tự động.
"""
import torch
import torch.nn as nn
import timm


class CNNBackbone(nn.Module):
    """
    EfficientNet-B4 pretrained trên ImageNet.
    Theo bài báo: 20 lớp đầu được đóng băng, chỉ fine-tune phần còn lại.
    Các MBConvBlock tích hợp sẵn Squeeze-and-Excitation (một dạng Channel Attention).
    """
    def __init__(self, freeze_layers=20):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            features_only=True
        )
        # Đóng băng 20 lớp đầu theo đúng bài báo
        layers = list(self.model.children())
        # Lưu ý: timm model.children() có thể không chia đúng từng "lớp" theo ý người dùng,
        # nên đóng băng theo parameters là an toàn nhất.
        for i, param in enumerate(self.model.parameters()):
            if i < freeze_layers:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)[-1]


class ViTBlock(nn.Module):
    """
    Vision Transformer Block theo thiết kế của bài báo.
    Cấu hình tốt nhất theo bài báo là ViT-Large-Patch16 (d_model=1024, nhead=16)
    nhưng để vừa với GPU, ta dùng d_model mặc định bằng chiều của CNN (1792) và nhead=8.

    KHÔNG có Positional Encoding — đây là thiết kế CÓ CHỦ Ý:
    Tác giả lập luận rằng feature map từ EfficientNet đã mã hóa sẵn
    thông tin vị trí không gian (spatial) và ngữ nghĩa (semantic).
    Do đó, không cần bổ sung Positional Encoding thêm.
    """
    def __init__(self, dim, num_layers=2):
        super().__init__()
        # Bài báo yêu cầu Dropout 0.3 trước biến đổi tuyến tính
        self.dropout = nn.Dropout(0.3)
        self.proj = nn.Linear(dim, dim)
        
        # Transformer Encoder Block theo công thức của bài báo: 
        # Z' = MHSA(LN(Z)) + Z; Z = FFN(LN(Z')) + Z'
        enc = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN chuẩn của ViT
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x):
        # x: (B, C, H, W) từ CNN
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C) — spatial tokens
        
        # Fusion theo bài báo
        x = self.dropout(x)
        x = self.proj(x)
        
        # Đi qua Transformer
        x = self.encoder(x)               # Self-attention học vùng bệnh
        
        # Lấy token trung bình làm representation cuối cùng (Global Average Pooling)
        return x.mean(dim=1)              # (B, C)


class Model(nn.Module):
    """
    Kiến trúc đầy đủ: EfficientNet-B4 (freeze 20 lớp) → Dropout(0.3) → Linear → ViT → Classifier
    Theo đúng mô tả trong bài báo.
    """
    def __init__(self, num_classes, freeze_layers=20):
        super().__init__()
        self.cnn = CNNBackbone(freeze_layers=freeze_layers)
        with torch.no_grad():
            c = self.cnn(torch.randn(1, 3, 224, 224)).shape[1]
        
        self.vit = ViTBlock(c)
        # Dense -> Softmax (Bài báo dùng Softmax nhưng ta trả về raw logits vì CrossEntropyLoss tự áp dụng Softmax)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.vit(x)
        return self.classifier(x)
