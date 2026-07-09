"""
model.py - Định nghĩa kiến trúc Model dùng chung cho train và test
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


class SwinBlockWrapper(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=[7, 7]):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        shift = [window_size[0] // 2, window_size[1] // 2]

        self.block1 = SwinTransformerBlock(
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=[0, 0], mlp_ratio=4.0, dropout=0.1,
            attention_dropout=0.1, norm_layer=nn.LayerNorm
        )
        self.block2 = SwinTransformerBlock(
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=shift, mlp_ratio=4.0, dropout=0.1,
            attention_dropout=0.1, norm_layer=nn.LayerNorm
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.proj(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean(dim=[1, 2])     # (B, C)
        return x


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
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class DAAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class Model(nn.Module):
    def __init__(self, num_classes):
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
