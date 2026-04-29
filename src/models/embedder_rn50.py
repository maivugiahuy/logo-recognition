"""
ResNet50 embedder variant — paper ablation (Table 3 ResNet50 row).
  - torchvision ResNet50 pretrained on ImageNet1K
  - Final avgpool replaced with adaptive max pool
  - Layer norm before output
  - FC 2048→128, L2-normalized
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ResNet50Embedder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        # Replace avgpool with adaptive max pool (Sec 4.2)
        base.avgpool = nn.AdaptiveMaxPool2d(1)
        # Remove original fc
        self.trunk = nn.Sequential(*list(base.children())[:-1])  # → (B, 2048, 1, 1)
        self.ln = nn.LayerNorm(2048)
        self.fc = nn.Linear(2048, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x).flatten(1)  # (B, 2048)
        feat = self.ln(feat)
        out = self.fc(feat)
        return F.normalize(out, dim=-1)


def build_rn50_embedder(embed_dim: int = 128) -> ResNet50Embedder:
    return ResNet50Embedder(embed_dim=embed_dim)
