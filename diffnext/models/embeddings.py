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
"""Embedding layers."""

from typing import Tuple

import torch
from torch import nn


class PosEmbed(nn.Module):
    """Position embedding layer."""

    def __init__(self, dim, base_size=(16, 16)):
        super(PosEmbed, self).__init__()
        (self.base_h, self.base_w), self.space_embed = base_size, None
        self.freq_hw = 1 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)))

    def get_space_embed(self, device=None, dtype=None) -> torch.Tensor:
        h, w = self.base_h, self.base_w
        if self.space_embed is not None and self.space_embed.size(0) == h * w:
            return self.space_embed
        grid_h = torch.arange(h, dtype=torch.float32) * (self.base_h / h)
        grid_w = torch.arange(w, dtype=torch.float32) * (self.base_w / w)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")
        freq_w, freq_h = [_.reshape(-1, 1) * self.freq_hw.unsqueeze(0) for _ in (grid_w, grid_h)]
        embed = torch.cat([freq_w.sin(), freq_w.cos(), freq_h.sin(), freq_h.cos()], dim=-1)
        self.space_embed = embed.to(device=device, dtype=dtype)
        return self.space_embed

    def forward(self, x) -> torch.Tensor:
        return x.add_(self.get_space_embed(x.device, x.dtype))


class VideoPosEmbed(PosEmbed):
    """Video position embedding layer."""

    def __init__(self, dim, base_size):
        super(VideoPosEmbed, self).__init__(dim, base_size=base_size[1:])
        self.base_t, self.time_embed, self.norm = base_size[0], None, nn.LayerNorm(dim)
        self.time_proj = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.freq_t = 1 / (10000 ** (torch.arange(128, dtype=torch.float32).unsqueeze(0) / 128))

    def get_time_embed(self, t) -> torch.Tensor:
        if self.time_embed is not None and t == self.time_embed.size(0):
            return self.norm(self.time_proj(self.time_embed))
        device, dtype = self.time_proj[0].weight.device, self.time_proj[0].weight.dtype
        grid = torch.arange(t, dtype=torch.float32) / (t / self.base_t)
        freq_t = grid.view(-1, 1, 1).mul(self.freq_t)
        sincos = torch.cat([freq_t.sin(), freq_t.cos()], dim=-1)
        self.time_embed = sincos.to(device=device, dtype=dtype)
        return self.norm(self.time_proj(self.time_embed))

    def forward(self, x) -> torch.Tensor:
        x = x.add_(self.get_time_embed(x.size(-3))) if x.dim() == 4 else x
        return x.add_(self.get_space_embed(x.device, x.dtype))


class MotionEmbed(nn.Module):
    """Motion embedding layer."""

    def __init__(self, dim, base_flow=5, base_fps=12):
        super(MotionEmbed, self).__init__()
        self.base_flow, self.base_fps = base_flow, base_fps
        self.flow_proj = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.fps_proj = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.freq_m = 1 / (10000 ** (torch.arange(128, dtype=torch.float32).unsqueeze(0) / 128))

    def get_embed(self, c, x, k) -> torch.Tensor:
        x = [getattr(self, f"base_{k}")] * c.size(0) if x is None else x
        freq_m = torch.as_tensor(x).view(-1, 1, 1).float().mul(self.freq_m)
        sincos = torch.cat([freq_m.sin(), freq_m.cos()], dim=-1)
        return getattr(self, f"{k}_proj")(sincos.to(device=c.device, dtype=c.dtype))

    def forward(self, c, flow=None, fps=None) -> torch.Tensor:
        outputs = [self.get_embed(c, x, k) for k, x in [("flow", flow), ("fps", fps)]]
        return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]


class PatchEmbed(nn.Module):
    """Patch embedding layer."""

    def __init__(self, image_dim, embed_dim, patch_size):
        super(PatchEmbed, self).__init__()
        self.height = self.width = None
        self.image_dim, self.patch_size = image_dim, patch_size
        self.proj = nn.Conv2d(image_dim, embed_dim, patch_size, patch_size)

    def patchify(self, x) -> torch.Tensor:
        x = x.view(-1, self.image_dim, self.height, self.patch_size, self.width, self.patch_size)
        return x.permute(0, 2, 4, 3, 5, 1).flatten(1, 2).flatten(2, 4).contiguous()

    def unpatchify(self, x) -> torch.Tensor:
        x = x.view(-1, self.height, self.width, self.patch_size, self.patch_size, self.image_dim)
        return x.permute(0, 5, 1, 3, 2, 4).flatten(2, 3).flatten(3, 4).contiguous()

    def forward(self, x) -> torch.Tensor:
        flat_shape = (x.size(0), x.size(2)) if x.dim() == 5 else None
        x = x.transpose(1, 2).flatten(0, 1) if x.dim() == 5 else x
        self.width = x.size(-1) // self.patch_size if x.dim() == 4 else self.width
        self.height = x.size(-2) // self.patch_size if x.dim() == 4 else self.height
        x = self.proj(x).flatten(2).transpose(1, 2) if x.dim() == 4 else x
        return x.view(flat_shape + x.shape[1:]) if flat_shape else x


class TextEmbed(nn.Module):
    """Encode text tokens into embeddings."""

    def __init__(self, token_dim, embed_dim, num_tokens=256):
        super(TextEmbed, self).__init__()
        self.token_dim, self.num_tokens, self.encoders = token_dim, num_tokens, []
        self.proj, self.norm = nn.Linear(token_dim, embed_dim), nn.LayerNorm(embed_dim)
        self.register_buffer("weight", torch.zeros(512, token_dim))  # Maximum positions.
        nn.init.normal_(self.weight, std=0.02)

    def add_guidance(self, c) -> torch.Tensor:
        token = self.weight[: self.num_tokens].expand(c.size(0), -1, -1)
        return torch.cat([c, self.forward(token)])

    @torch.no_grad()
    def encode_prompts(self, prompts) -> torch.Tensor:
        device, dtype = self.weight.device, self.weight.dtype
        x = self.weight[: self.num_tokens].expand(len(prompts), -1, -1).clone()
        for i, p in enumerate(prompts if not isinstance(prompts[0], str) else []):
            x[i, : p.shape[0]] = torch.as_tensor(p, device=device).to(dtype)
        if not isinstance(prompts[0], str):
            return x
        tokenizer, encoder = self.encoders
        trunc_args = {"max_length": self.num_tokens, "truncation": True}
        pad_args = {"padding": "max_length", **trunc_args}
        tokens = [tokenizer(p, **pad_args).input_ids for p in prompts]
        maxlens = [len(tokenizer(p, **trunc_args).input_ids) for p in prompts]
        tokens = torch.as_tensor(tokens, device=encoder.device)
        embeds = encoder(tokens).last_hidden_state.to(dtype=dtype)
        for i, maxlen in enumerate(maxlens):
            x[i, :maxlen] = embeds[i, :maxlen]
        return x

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            return self.norm(self.proj(self.encode_prompts(x)))
        return self.norm(self.proj(x))


class MaskEmbed(nn.Module):
    """Apply mask positions to input embeddings."""

    def __init__(self, embed_dim, mask_ratios=(0.7, 1.0)):
        super(MaskEmbed, self).__init__()
        self.mask_ratios = mask_ratios
        self.bos_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        [nn.init.normal_(_, std=0.02) for _ in (self.bos_token, self.mask_token)]
        self.mask, self.attn_mask = None, None
        self.pred_ids, self.pred_pos, self.generator = None, 0, None

    def get_pred_mask(self, num_preds) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the current mask for next prediction."""
        if self.pred_ids is None:
            u_dist = torch.empty_like(self.mask).uniform_(generator=self.generator)
            self.pred_ids = u_dist.argsort(dim=1)
        pred_ids = self.pred_ids[:, self.pred_pos : self.pred_pos + num_preds]
        pred_mask = torch.zeros_like(self.mask).scatter_(1, pred_ids, 1)
        self.pred_pos, self.mask = self.pred_pos + num_preds, self.mask.mul_(1 - pred_mask)
        return pred_mask, pred_ids

    def apply_mask(self, x) -> torch.Tensor:
        """Apply the current mask to input."""
        return x.mul(1 - self.mask).add_(self.mask_token * self.mask)

    def forward(self, x) -> torch.Tensor:
        if self.mask is None:
            self.mask, self.pred_pos = x.new_ones(x.shape[:-1] + (1,)), 0
        return self.apply_mask(x)
