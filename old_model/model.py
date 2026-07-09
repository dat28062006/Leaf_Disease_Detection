"""
old_model/model.py - Kiến trúc gốc từ notebook (EfficientNet-B4 + ViT)
"""
import torch
import torch.nn as nn
import timm


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


class ViTBlock(nn.Module):
    """Vision Transformer block đơn giản (không có Positional Encoding - giữ nguyên bản gốc)."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        enc = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.proj(x)
        x = self.encoder(x)
        return x.mean(dim=1)              # (B, C)


class Model(nn.Module):
    """EfficientNet-B4 → ViT Block → Classifier"""
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNBackbone()
        with torch.no_grad():
            c = self.cnn(torch.randn(1, 3, 224, 224)).shape[1]
        self.vit        = ViTBlock(c)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.vit(x)
        return self.classifier(x)
