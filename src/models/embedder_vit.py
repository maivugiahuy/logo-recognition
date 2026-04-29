"""
ViT-B/32 embedder — paper Sec 4.2:
  - open_clip ViT-B/32 trunk with OpenAI pretrained weights
  - bicubic-interpolate positional embeddings 224→160
  - FC 768→128 head
  - L2-normalized output
"""
import math

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTEmbedder(nn.Module):
    def __init__(self, embed_dim: int = 128, input_size: int = 160):
        super().__init__()
        self.input_size = input_size

        # Load trunk (vision tower only)
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.trunk = clip_model.visual

        # Interpolate positional embeddings from 224→input_size
        if input_size != 224:
            self._interpolate_pos_embed(input_size)

        trunk_dim = self.trunk.output_dim  # 512 for ViT-B/32 (post-projection dim)

        # FC 512→128: paper Sec 4.2 "FC layer of shape 128 × d0,
        # where d0 is the output dimension of the base (trunk) architecture."
        # open_clip ViT-B/32 output_dim = 512 (sau CLIP projection layer).
        self.fc = nn.Linear(trunk_dim, embed_dim)

    def _interpolate_pos_embed(self, new_size: int) -> None:
        """Bicubic interpolate class-token + patch positional embeddings."""
        pos_embed = self.trunk.positional_embedding  # (1 + grid^2, D)
        patch_size = self.trunk.conv1.kernel_size[0]  # 32
        old_grid = int(math.sqrt(pos_embed.shape[0] - 1))  # 7 for 224/32
        new_grid = new_size // patch_size  # 5 for 160/32

        cls_tok = pos_embed[:1, :]
        patch_tokens = pos_embed[1:, :]  # (old_grid^2, D)
        D = patch_tokens.shape[1]

        # Reshape to spatial grid for bicubic interp
        patch_tokens = patch_tokens.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)  # (1,D,H,W)
        patch_tokens = F.interpolate(
            patch_tokens.float(), size=(new_grid, new_grid), mode="bicubic", align_corners=False
        ).to(pos_embed.dtype)
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(new_grid * new_grid, D)

        new_pos_embed = torch.cat([cls_tok, patch_tokens], dim=0)
        self.trunk.positional_embedding = nn.Parameter(new_pos_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)  # (B, trunk_dim)
        out = self.fc(features)
        out = F.normalize(out, dim=-1)
        return out


def build_vit_embedder(
    embed_dim: int = 128,
    input_size: int = 160,
    freeze_blocks: int = 8,  # freeze first N of 12 transformer blocks (~60% less backward)
) -> ViTEmbedder:
    model = ViTEmbedder(embed_dim=embed_dim, input_size=input_size)
    if freeze_blocks > 0:
        blocks = model.trunk.transformer.resblocks
        for i, block in enumerate(blocks):
            if i < freeze_blocks:
                for p in block.parameters():
                    p.requires_grad = False
        frozen = sum(1 for p in model.trunk.parameters() if not p.requires_grad)
        total = sum(1 for p in model.trunk.parameters())
        print(f"Frozen {freeze_blocks}/{len(blocks)} ViT blocks ({frozen}/{total} trunk params frozen)")
    return model
