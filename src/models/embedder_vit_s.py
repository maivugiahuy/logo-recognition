"""ViT-S/16 student embedder for knowledge distillation (via timm)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

TRUNK_DIM = 384  # ViT-S/16 CLS token dim


class VitSEmbedder(nn.Module):
    def __init__(self, embed_dim: int = 128, input_size: int = 160):
        super().__init__()
        import timm
        # img_size triggers positional embedding interpolation from native 224
        self.trunk = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
            img_size=input_size,
        )
        self.fc = nn.Linear(TRUNK_DIM, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(self.trunk(x)), dim=-1)


def build_vit_s_embedder(embed_dim: int = 128, input_size: int = 160) -> VitSEmbedder:
    return VitSEmbedder(embed_dim=embed_dim, input_size=input_size)
