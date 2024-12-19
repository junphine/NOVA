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
"""Base 3D transformer model for video generation."""

from typing import Dict

import torch
from torch import nn
from tqdm import tqdm


class Transformer3DModel(nn.Module):
    """Base 3D transformer model for video generation."""

    def __init__(
        self,
        video_encoder=None,
        image_encoder=None,
        image_decoder=None,
        mask_embed=None,
        text_embed=None,
        video_pos_embed=None,
        motion_embed=None,
        sample_scheduler=None,
    ):
        super(Transformer3DModel, self).__init__()
        self.video_encoder = video_encoder
        self.image_encoder = image_encoder
        self.image_decoder = image_decoder
        self.mask_embed = mask_embed
        self.text_embed = text_embed
        self.video_pos_embed = video_pos_embed
        self.motion_embed = motion_embed
        self.sample_scheduler = sample_scheduler

    def preprocess(self, inputs: Dict):
        """Preprocess model inputs."""
        dtype, device = self.dtype, self.device
        inputs["c"], add_guidance = inputs.get("c", []), inputs.get("guidance_scale", 1) != 1
        if "x" not in inputs:
            batch_size = inputs.get("batch_size", 1)
            image_size = (self.image_encoder.image_dim,) + self.image_encoder.image_size
            inputs["x"] = torch.empty(batch_size, *image_size, device=device, dtype=dtype)
        if inputs.get("prompt", None) is not None and self.text_embed:
            inputs["c"].append(self.text_embed(inputs.pop("prompt")))
        if inputs.get("motion_flow", None) is not None and self.motion_embed:
            flow, fps = inputs.pop("motion_flow", None), inputs.pop("fps", None)
            flow, fps = [v + v if (add_guidance and v) else v for v in (flow, fps)]
            inputs["c"].append(self.motion_embed(inputs["c"][-1], flow, fps))
        inputs["c"] = torch.cat(inputs["c"], dim=1) if len(inputs["c"]) > 1 else inputs["c"][0]

    @torch.no_grad()
    def postprocess(self, outputs: Dict, inputs: Dict):
        """Postprocess model outputs."""
        if inputs.get("output_type", "np") == "latent":
            return outputs
        x = inputs["vae"].unscale_(outputs.pop("x"))
        batch_size, vae_batch_size = x.size(0), inputs.get("vae_batch_size", 1)
        sizes, splits = [vae_batch_size] * (batch_size // vae_batch_size), []
        sizes += [batch_size - sum(sizes)] if sum(sizes) != batch_size else []
        for x_split in x.split(sizes) if len(sizes) > 1 else [x]:
            splits.append(inputs["vae"].decode(x_split).sample)
        x = torch.cat(splits) if len(splits) > 1 else splits[0]
        x = x.permute(0, 2, 3, 4, 1) if x.dim() == 5 else x.permute(0, 2, 3, 1)
        outputs["x"] = x.mul_(127.5).add_(127.5).clamp(0, 255).byte()

    def progress_bar(self, iterable, enable=True):
        """Return a tqdm progress bar."""
        return tqdm(iterable) if enable else iterable

    @torch.no_grad()
    def denoise(self, z, x, guidance_scale=1, generator=None, pred_ids=None) -> torch.Tensor:
        """Run diffusion denoising process."""
        self.sample_scheduler._step_index = None  # Reset counter.
        for t in self.sample_scheduler.timesteps:
            x_pack = torch.cat([x] * 2) if guidance_scale > 1 else x
            timestep = torch.as_tensor(t, device=x.device).expand(z.shape[0])
            noise_pred = self.image_decoder(x_pack, timestep, z, pred_ids)
            if guidance_scale > 1:
                cond, uncond = noise_pred.chunk(2)
                noise_pred = uncond.add_(cond.sub_(uncond).mul_(guidance_scale))
            noise_pred = self.image_encoder.patch_embed.unpatchify(noise_pred)
            x = self.sample_scheduler.step(noise_pred, t, x, generator=generator).prev_sample
        return self.image_encoder.patch_embed.patchify(x)

    @torch.inference_mode()
    def generate_frame(self, states: Dict, inputs: Dict):
        """Generate a batch of frames."""
        guidance_scale = inputs.get("guidance_scale", 1)
        min_guidance_scale = inputs.get("min_guidance_scale", guidance_scale)
        max_guidance_scale = inputs.get("max_guidance_scale", guidance_scale)
        generator = self.mask_embed.generator = inputs.get("generator", None)
        all_num_preds = [_ for _ in inputs["num_preds"] if _ > 0]
        guidance_end = max_guidance_scale if states["t"] else guidance_scale
        guidance_start = max_guidance_scale if states["t"] else min_guidance_scale
        c, x, self.mask_embed.mask = states["c"], states["x"].zero_(), None
        for i, num_preds in enumerate(self.progress_bar(all_num_preds, inputs.get("tqdm2", False))):
            guidance_level = (i + 1) / len(all_num_preds)
            guidance_scale = (guidance_end - guidance_start) * guidance_level + guidance_start
            z = self.mask_embed(self.image_encoder.patch_embed(x))
            pred_mask, pred_ids = self.mask_embed.get_pred_mask(num_preds)
            pred_ids = torch.cat([pred_ids] * 2) if guidance_scale > 1 else pred_ids
            prev_ids = prev_ids if i else pred_ids.new_empty((pred_ids.size(0), 0, 1))
            z = torch.cat([z] * 2) if guidance_scale > 1 else z
            z = self.image_encoder(z, c, prev_ids)
            prev_ids = torch.cat([prev_ids, pred_ids], dim=1)
            states["noise"].normal_(generator=generator)
            sample = self.denoise(z, states["noise"], guidance_scale, generator, pred_ids)
            x.add_(self.image_encoder.patch_embed.unpatchify(sample.mul_(pred_mask)))

    @torch.inference_mode()
    def generate_video(self, inputs: Dict):
        """Generate a batch of videos."""
        guidance_scale = inputs.get("guidance_scale", 1)
        max_latent_length = inputs.get("max_latent_length", 1)
        self.sample_scheduler.set_timesteps(inputs.get("num_diffusion_steps", 25))
        states = {"x": inputs["x"], "noise": inputs["x"].clone()}
        latents, self.mask_embed.pred_ids = inputs.get("latents", []), None
        [setattr(blk.attn, "cache_kv", max_latent_length > 1) for blk in self.video_encoder.blocks]
        time_embed = self.video_pos_embed.get_time_embed(max_latent_length)
        for states["t"] in self.progress_bar(range(max_latent_length), inputs.get("tqdm1", True)):
            c = self.video_encoder.patch_embed(states["x"])
            c.__setitem__(slice(None), self.mask_embed.bos_token) if states["t"] == 0 else c
            c = self.video_pos_embed(c.add_(time_embed[states["t"]]))
            c = torch.cat([c] * 2) if guidance_scale > 1 else c
            c = self.video_encoder(c, None if states["t"] else inputs["c"])
            states["c"] = self.video_encoder.mixer(states["*"], c) if states["t"] else c
            states["*"] = states["*"] if states["t"] else states["c"]
            if states["t"] == 0 and latents:
                states["x"].copy_(latents[-1])
            else:
                self.generate_frame(states, inputs)
                latents.append(states["x"].clone())
        [setattr(blk.attn, "cache_kv", False) for blk in self.video_encoder.blocks]

    def forward(self, inputs: Dict) -> Dict:
        """Define the computation performed at every call."""
        self.preprocess(inputs)
        inputs["latents"] = inputs.pop("latents", [])
        self.generate_video(inputs)
        outputs = {"x": torch.stack(inputs["latents"], dim=2)}
        self.postprocess(outputs, inputs)
        return outputs
