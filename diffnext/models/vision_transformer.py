# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Vision Transformer."""

import torch
from torch import nn

from diffnext.models.embeddings import PatchEmbed


class MLP(nn.Module):
    """Two layers MLP."""

    def __init__(self, dim, mlp_ratio=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.activation = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class Attention(nn.Module):
    """Multihead attention."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super(Attention, self).__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_mask, self.cache_kv = None, None

    def forward(self, x) -> torch.Tensor:
        qkv_shape = [-1, x.size(1), 3, self.num_heads, self.head_dim]
        q, k, v = self.qkv(x).view(qkv_shape).permute(2, 0, 3, 1, 4).unbind(dim=0)
        if self.cache_kv is not None:
            if isinstance(self.cache_kv, list):
                k = self.cache_kv[0] = torch.cat([self.cache_kv[0], k], dim=2)
                v = self.cache_kv[1] = torch.cat([self.cache_kv[1], v], dim=2)
            else:
                self.cache_kv = [k, v]
        o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask)
        return self.proj(o.transpose(1, 2).flatten(2))


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=True):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)

    def forward(self, x) -> torch.Tensor:
        x = self.norm1(self.attn(x)).add_(x)
        return self.norm2(self.mlp(x)).add_(x)


class VisionTransformer(nn.Module):
    """Vision transformer."""

    def __init__(
        self,
        depth,
        embed_dim,
        num_heads,
        mlp_ratio=4,
        patch_size=2,
        image_size=32,
        image_dim=4,
        encoder_depth=None,
    ):
        super(VisionTransformer, self).__init__()
        self.embed_dim, self.image_size, self.image_dim = embed_dim, image_size, image_dim
        self.patch_embed = PatchEmbed(image_dim, embed_dim, patch_size)
        self.pos_embed = nn.Identity()
        self.blocks = nn.ModuleList(Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth))
        self.norm, self.mixer = nn.LayerNorm(embed_dim), nn.Identity()
        self.encoder_depth = len(self.blocks) // 2 if encoder_depth is None else encoder_depth

    def forward(self, x, c=None, prev_ids=None) -> torch.Tensor:
        x, prev_ids = x if isinstance(x, (tuple, list)) else (x, prev_ids)
        prev_ids = prev_ids if self.encoder_depth else None
        x = x_masked = self.pos_embed(self.patch_embed(x))
        if prev_ids is not None:  # Split mask from x.
            prev_ids = prev_ids.expand(-1, -1, x.size(-1))
            x = x.gather(1, prev_ids)
        x = x if c is None else torch.cat([c, x], dim=1)
        for blk in self.blocks[: self.encoder_depth]:
            x = blk(x)
        if prev_ids is not None and c is not None:  # Split c from x.
            c, x = x.split((c.size(1), x.size(1) - c.size(1)), dim=1)
        if prev_ids is not None:  # Merge mask with x.
            x = x_masked.scatter(1, prev_ids, x)
            x = x if c is None else torch.cat([c, x], dim=1)
        for blk in self.blocks[self.encoder_depth :]:
            x = blk(x)
        return self.norm(x if c is None else x[:, c.size(1) :])