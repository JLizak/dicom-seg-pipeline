"""Fabryka architektur — nazwa w configu → sieć MONAI.

Dzięki temu porównanie SegResNet / DynUNet / SwinUNETR to zmiana jednej linijki
w YAML-u, a nie w kodzie.
"""

import torch.nn as nn
from monai.networks.nets import DynUNet, SegResNet, SwinUNETR, UNet


def build_model(name: str, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "segresnet":
        params = dict(init_filters=16, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))
        params.update(kwargs)
        return SegResNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels, **params)
    if name == "unet":
        params = dict(channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
        params.update(kwargs)
        return UNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels, **params)
    if name == "dynunet":
        # Układ jak w nnU-Net: 5 poziomów, pierwszy bez downsamplingu.
        depth = kwargs.pop("depth", 5)
        strides = [[1, 1, 1]] + [[2, 2, 2]] * (depth - 1)
        params = dict(
            kernel_size=[[3, 3, 3]] * depth,
            strides=strides,
            upsample_kernel_size=strides[1:],
            res_block=True,
        )
        params.update(kwargs)
        return DynUNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels, **params)
    if name == "swinunetr":
        # Wymiary patcha muszą być podzielne przez 32.
        params = dict(feature_size=24)
        params.update(kwargs)
        return SwinUNETR(spatial_dims=3, in_channels=in_channels, out_channels=out_channels, **params)
    raise ValueError(f"Nieznana architektura: {name!r}")
