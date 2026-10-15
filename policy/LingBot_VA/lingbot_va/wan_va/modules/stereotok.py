"""LingBot codec surface backed by the existing frozen StereoTok student."""
from types import SimpleNamespace

import torch
from fastwam.models.wan22.stereotok.vae2_2 import latent_scale
from fastwam.models.wan22.stereotok_vae import StereoTokVAE


class StereoTokAutoencoder(StereoTokVAE):
    def __init__(self, checkpoint, device):
        super().__init__(checkpoint, device)
        if self.binding['sha256'] != '61c6057b49bec9c52f96793a20247d3746978939f0551c37142bdbda6b252eae':
            raise ValueError('LingBot StereoTok requires the selected u15853 export')
        mean, inverse_std = latent_scale('cpu')
        self.config = SimpleNamespace(z_dim=48, patch_size=2,
                                      latents_mean=mean.tolist(), latents_std=inverse_std.reciprocal().tolist())

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def decode(self, latents, return_dict=False):
        if return_dict:
            raise ValueError('LingBot codec uses tuple decode output')
        # LingBot has already inverted normalization before calling its VAE.
        with torch.autocast(device_type=latents.device.type, enabled=False):
            rgb, _ = self.model.decode(latents.float(), (0., 1.))
        return (rgb,)
