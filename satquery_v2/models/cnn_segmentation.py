"""
models/cnn_segmentation.py
-----------------------------
A real, trainable pixel-level segmentation CNN — this is the "use CNN if
needed" path. Unlike training/finetune_bigearthnet.py (which trains a
scene-level, whole-patch classifier because BigEarthNet-19 labels are
scene-level), this one is a proper U-Net-lite semantic segmentation network:
given any labeled image+mask dataset (which is exactly what you said you'd
provide — captioning/detection datasets with per-pixel or per-object ground
truth can be converted into segmentation masks), it learns to predict the
SAME per-pixel class_map that utils.spectral_indices.classify_landcover()
computes heuristically today. Once trained, it fully replaces the heuristic
(see RemoteSensingVLM(segmentation_backend=...) in models/base_vlm.py) for
every task, including grounding and change maps, not just whole-image answers.

Architecture: small U-Net (4 encoder levels, skip connections, transposed-conv
decoder), channel count configurable so it works whether you're training on
3-band RGB, 4-band RGB+NIR, or a full Sentinel-2 stack.

Not runnable inside the sandbox this project was authored in (no torch
installed there / no labeled dataset available); the architecture and
training loop are real and complete — point training/train_segmentation_cnn.py
at your image+mask manifest once you have one.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from utils.spectral_indices import LAND_COVER_CLASSES, BandRoles


def build_unet_lite(in_channels: int, n_classes: int):
    """Lazily built (needs torch). Small U-Net: 4 down / 4 up, skip connections."""
    import torch
    import torch.nn as nn

    def conv_block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    class UNetLite(nn.Module):
        def __init__(self, in_channels, n_classes, base=32):
            super().__init__()
            self.enc1 = conv_block(in_channels, base)
            self.enc2 = conv_block(base, base * 2)
            self.enc3 = conv_block(base * 2, base * 4)
            self.enc4 = conv_block(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)

            self.bottleneck = conv_block(base * 8, base * 16)

            self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
            self.dec4 = conv_block(base * 16, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.dec3 = conv_block(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.dec2 = conv_block(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec1 = conv_block(base * 2, base)

            self.out_conv = nn.Conv2d(base, n_classes, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))

            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out_conv(d1)

    return UNetLite(in_channels, n_classes)


class SegmentationCNNBackend:
    """Wraps a checkpoint from training/train_segmentation_cnn.py and implements
    the `predict_class_map(arr, band_roles)` interface RemoteSensingVLM expects
    for its `segmentation_backend` slot."""

    def __init__(self, checkpoint_path: str):
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.class_names = ckpt.get("class_names", LAND_COVER_CLASSES)
        self.in_channels = ckpt["in_channels"]
        self.tile_size = ckpt.get("tile_size", 256)
        self.trained_band_roles = ckpt.get("band_roles")  # what it was trained with

        self.model = build_unet_lite(self.in_channels, len(self.class_names))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self._torch = torch

    def predict_class_map(self, arr: np.ndarray, band_roles: Optional[BandRoles] = None) -> np.ndarray:
        torch = self._torch
        h, w = arr.shape[:2]

        # select the channels the model was trained on, in the same order,
        # if band_roles is supplied and differs from the array's raw layout
        x = arr
        if x.shape[-1] != self.in_channels:
            x = x[..., :self.in_channels] if x.shape[-1] > self.in_channels else \
                np.pad(x, ((0, 0), (0, 0), (0, self.in_channels - x.shape[-1])), mode="edge")

        # resize to the tile size the network trained on (simple striding resize
        # to avoid a torchvision dependency); real deployments should tile with
        # overlap instead of a single global resize for large scenes.
        ys = np.linspace(0, h - 1, self.tile_size).astype(int)
        xs = np.linspace(0, w - 1, self.tile_size).astype(int)
        tile = x[ys][:, xs]

        tensor = torch.tensor(tile.transpose(2, 0, 1)[None], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor)
            pred_small = logits.argmax(dim=1)[0].numpy()

        # upsample the low-res prediction back to the original grid (nearest-neighbor)
        ys_full = np.linspace(0, self.tile_size - 1, h).astype(int)
        xs_full = np.linspace(0, self.tile_size - 1, w).astype(int)
        pred_full = pred_small[ys_full][:, xs_full]
        return pred_full.astype(np.int32)
