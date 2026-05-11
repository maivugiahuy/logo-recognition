"""
DINOv2-B/14 embedder:
  - facebookresearch/dinov2 ViT-B/14 pretrained trunk (torch.hub)
  - CLS token output 768-d → FC 768→128
  - L2-normalized output
  - Native input size 224×224 (patch 14, 16×16 grid)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

TRUNK_DIM = 768  # ViT-B/14 CLS token output dim


class DINOv2Embedder(nn.Module):
    def __init__(self, embed_dim: int = 128, input_size: int = 224):
        super().__init__()
        self.input_size = input_size
        self.trunk = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
        )
        self.fc = nn.Linear(TRUNK_DIM, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)  # (B, 768) — CLS token, L2-normalized by DINOv2
        out = self.fc(features)
        return F.normalize(out, dim=-1)


def build_dinov2_embedder(
    embed_dim: int = 128,
    input_size: int = 224,
    freeze_blocks: int = 0,
) -> DINOv2Embedder:
    model = DINOv2Embedder(embed_dim=embed_dim, input_size=input_size)
    if freeze_blocks > 0:
        blocks = model.trunk.blocks
        for i, block in enumerate(blocks):
            if i < freeze_blocks:
                for p in block.parameters():
                    p.requires_grad = False
        frozen = sum(1 for p in model.trunk.parameters() if not p.requires_grad)
        total = sum(1 for p in model.trunk.parameters())
        print(f"Frozen {freeze_blocks}/{len(blocks)} DINOv2 blocks "
              f"({frozen}/{total} trunk params frozen)")
    return model
