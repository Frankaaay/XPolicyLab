# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
    stereotok_checkpoint=None,
    stereotok_input_mode="stereo",
):
    if stereotok_checkpoint is not None:
        from .stereotok import StereoTokAutoencoder
        return StereoTokAutoencoder(stereotok_checkpoint, torch_device, input_mode=stereotok_input_mode)
    if stereotok_input_mode != "stereo":
        raise ValueError('A StereoTok input mode requires its checkpoint')
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    **kwargs
):
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs
    )
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.stereo = bool(getattr(vae_model, 'requires_stereo', False))
        self.stereotok = getattr(vae_model, 'binding', {}).get('format') == 'stereotok-student-v3'
        if self.stereotok:
            from fastwam.models.wan22.stereotok.vae2_2 import count_conv3d
            self.encoder = vae_model.model.encoder
            self.quant_conv = vae_model.model.conv1
            self.enc_conv_num = count_conv3d(self.encoder)
            self.geometry_conv_num = count_conv3d(vae_model.model.fusion.downsamples)
            self.clear_cache()
            return
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num
        if self.stereotok:
            self.geometry_cache = [None] * self.geometry_conv_num
            self.first_chunk = True
            self.stream_shape = None

    @torch.no_grad()
    def encode_chunk(self, x_chunk, right_chunk=None):
        if self.stereotok and x_chunk.ndim == 5:
            expected = 1 if self.first_chunk else 4
            frames = x_chunk.shape[2]
            if frames > expected:
                if (frames - expected) % 4:
                    raise ValueError('StereoTok stream requires complete four-frame groups')
                if self.stereo and (right_chunk is None or right_chunk.shape != x_chunk.shape):
                    raise ValueError('StereoTok streaming requires matching real [B,3,T,H,W] eyes')
                if not self.stereo and right_chunk is not None:
                    raise ValueError('Mono-left StereoTok must not receive right eyes')
                for pixels in (x_chunk, right_chunk):
                    if pixels is not None and (not torch.isfinite(pixels).all() or pixels.abs().max() > 1.001):
                        raise ValueError('StereoTok inputs must be finite and normalized to [-1,1]')
                chunks = []
                start = 0
                while start < frames:
                    end = start + (1 if self.first_chunk else 4)
                    chunks.append(self.encode_chunk(x_chunk[:, :, start:end],
                        None if right_chunk is None else right_chunk[:, :, start:end]))
                    start = end
                return torch.cat(chunks, dim=2)
        if self.stereotok and not self.stereo:
            if right_chunk is not None:
                raise ValueError('Mono-left StereoTok must not receive right eyes')
            expected = 1 if self.first_chunk else 4
            if x_chunk.ndim != 5 or x_chunk.shape[1:3] != (3, expected):
                raise ValueError(f'Mono-left StereoTok stream needs [B,3,{expected},H,W]')
            shape = (x_chunk.shape[0], *x_chunk.shape[-2:])
            if any(n % 16 for n in x_chunk.shape[-2:]):
                raise ValueError('StereoTok stream requires 16-aligned images')
            if self.stream_shape is not None and self.stream_shape != shape:
                raise ValueError('Reset the stream before changing camera batch or resolution')
            if not torch.isfinite(x_chunk).all() or x_chunk.abs().max() > 1.001:
                raise ValueError('StereoTok inputs must be finite and normalized to [-1,1]')
            with torch.autocast(device_type=x_chunk.device.type, enabled=False):
                out = self.encoder(patchify(x_chunk.float(), 2), feat_cache=self.feat_cache, feat_idx=[0])
                result = self.quant_conv(out)
            self.first_chunk = False
            self.stream_shape = shape
            return result
        if self.stereo:
            if right_chunk is None or x_chunk.shape != right_chunk.shape or x_chunk.ndim != 5:
                raise ValueError('StereoTok streaming requires matching real [B,3,T,H,W] eyes')
            expected = 1 if self.first_chunk else 4
            if x_chunk.shape[1] != 3 or x_chunk.shape[2] != expected:
                raise ValueError(f'StereoTok stream needs {expected} frames in this chunk')
            shape = (x_chunk.shape[0], *x_chunk.shape[-2:])
            if self.stream_shape is not None and shape != self.stream_shape:
                raise ValueError('Reset the stream before changing camera batch or resolution')
            if not torch.isfinite(x_chunk).all() or not torch.isfinite(right_chunk).all():
                raise ValueError('Nonfinite stereo pixels')
            if x_chunk.abs().max() > 1.001 or right_chunk.abs().max() > 1.001:
                raise ValueError('StereoTok inputs must be normalized to [-1,1]')
            model = self.vae.model
            left, right = x_chunk.float(), right_chunk.float()
            mask = torch.ones(left.shape[0], 2, *left.shape[-2:], dtype=torch.bool, device=left.device)
            with torch.autocast(device_type=left.device.type, enabled=False):
                probabilities = model.fusion.prepare(left, right, mask, model.fusion.max_disparity)
                x = patchify(torch.cat((left, right), dim=0), 2)
                out = self.encoder(x, feat_cache=self.feat_cache, feat_idx=[0],
                    fusion=model.fusion, content_mask=mask, disparity=model.fusion.max_disparity,
                    match_probabilities=probabilities, geometry_cache=self.geometry_cache)
                result = self.quant_conv(out)
            self.first_chunk = False
            self.stream_shape = shape
            return result
        if right_chunk is not None:
            raise ValueError('Right-eye input supplied to native LingBot VAE')
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
