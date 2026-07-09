"""
new_model/model.py - Kiến trúc nâng cấp (EfficientNet-B4 + DAAM + Swin Transformer)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models.swin_transformer import SwinTransformerBlock

IMG_SIZE = 224


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
        avg    = torch.mean(x, dim=1, keepdim=True)
        mx, _  = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class DAAM(nn.Module):
    """Disease Attention Activation Module: kết hợp Channel + Spatial Attention."""
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
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        x = self.proj(x)
        x = self.block1(x)
        x = self.block2(x)
        return x.mean(dim=[1, 2])    # (B, C)


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
