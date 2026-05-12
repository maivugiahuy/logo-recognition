"""
DINOv3 ViT-B/16 embedder (timm vit_base_patch16_dinov3.lvd1689m):
  - Self-supervised on 1.689B images (LVD dataset)
  - CLS token 768-d → FC 768→128 → L2-norm
  - patch/16 at 160px → 10×10 = 100 tokens (vs DINOv2 B/14 at 168px → 144 tokens)
  - ImageNet normalization (same as DINOv2, different from CLIP)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

TRUNK_DIM = 768  # ViT-Base CLS token dim


class DINOv3Embedder(nn.Module):
    def __init__(self, embed_dim: int = 128, input_size: int = 160):
        super().__init__()
        import timm
        self.trunk = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m",
            pretrained=True,
            num_classes=0,
            img_size=input_size,
        )
        self.fc = nn.Linear(TRUNK_DIM, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(self.trunk(x)), dim=-1)


def build_dinov3_embedder(
    embed_dim: int = 128,
    input_size: int = 160,
    freeze_blocks: int = 0,
) -> DINOv3Embedder:
    model = DINOv3Embedder(embed_dim=embed_dim, input_size=input_size)
    if freeze_blocks > 0:
        blocks = model.trunk.blocks
        for i, block in enumerate(blocks):
            if i < freeze_blocks:
                for p in block.parameters():
                    p.requires_grad = False
    return model
