"""MinkUNet sparse 3D U-Net backbone for TARL pretrained weight loading.

This module adapts the MinkUNet architecture from TARL/SegContrast for use
within the OpenPCDet detection pipeline. The backbone produces 96-dim
point-wise features compatible with PointRCNN detection heads.

Architecture: stem -> stage1-4 (encoder) -> up1-4 (decoder) with skip connections.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np

from .smlp import SMLP
from ...utils import common_utils

__all__ = ["MinkUNet", "MinkUNetSMLP"]


class BasicConvolutionBlock(nn.Module):
    """Single sparse 3D convolution + batch norm + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        dimension: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                dimension=dimension,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        return self.net(x)


class BasicDeconvolutionBlock(nn.Module):
    """Single sparse 3D transposed convolution + batch norm + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dimension: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dimension=dimension,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """Sparse 3D residual block with optional downsampling shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        dimension: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                dimension=dimension,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=1,
                dimension=dimension,
            ),
            ME.MinkowskiBatchNorm(out_channels),
        )

        if in_channels == out_channels and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                    dimension=dimension,
                ),
                ME.MinkowskiBatchNorm(out_channels),
            )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        return self.relu(self.net(x) + self.downsample(x))


def _build_encoder_stage(
    in_channels: int,
    out_channels: int,
    dimension: int,
) -> nn.Sequential:
    """Build a single encoder stage: downsample + 2x residual blocks."""
    return nn.Sequential(
        BasicConvolutionBlock(
            in_channels, in_channels, kernel_size=2, stride=2, dilation=1, dimension=dimension
        ),
        ResidualBlock(
            in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dimension=dimension
        ),
        ResidualBlock(
            out_channels, out_channels, kernel_size=3, stride=1, dilation=1, dimension=dimension
        ),
    )


def _build_decoder_stage(
    in_channels: int,
    skip_channels: int,
    out_channels: int,
    dimension: int,
) -> nn.ModuleList:
    """Build a single decoder stage: upsample + concat skip + 2x residual blocks."""
    return nn.ModuleList(
        [
            BasicDeconvolutionBlock(
                in_channels, out_channels, kernel_size=2, stride=2, dimension=dimension
            ),
            nn.Sequential(
                ResidualBlock(
                    out_channels + skip_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    dimension=dimension,
                ),
                ResidualBlock(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    dimension=dimension,
                ),
            ),
        ]
    )


def _initialize_batchnorm_weights(module: nn.Module) -> None:
    """Initialize BatchNorm1d layers with weight=1, bias=0."""
    for m in module.modules():
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def _decode_stage(
    up_module: nn.ModuleList,
    x: ME.SparseTensor,
    skip: ME.SparseTensor,
) -> ME.SparseTensor:
    """Run one decoder stage: upsample, concat skip connection, refine."""
    y = up_module[0](x)
    y = ME.cat(y, skip)
    return up_module[1](y)


class MinkUNet(nn.Module):
    """Sparse 3D U-Net backbone using MinkowskiEngine.

    Produces 96-dimensional point-wise features from voxelized point clouds.
    Compatible with TARL pretrained weights via strict state_dict loading.

    The ``**kwargs`` signature accepts the keyword arguments passed by
    OpenPCDet's ``build_backbone_3d()``: ``model_cfg``, ``input_channels``,
    ``grid_size``, ``voxel_size``, ``point_cloud_range``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        cr: float = kwargs.get("cr", 1.0)
        in_channels: int = kwargs.get("in_channels", 4)
        cs = [int(cr * x) for x in [32, 32, 64, 128, 256, 256, 128, 96, 96]]
        self.run_up: bool = kwargs.get("run_up", True)
        self.D: int = kwargs.get("D", 3)
        self.num_point_features: int = 96

        grid_size: np.ndarray = kwargs.get("grid_size")
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.model_cfg = kwargs.get("model_cfg")
        self.voxel_size = kwargs.get("voxel_size")
        self.point_cloud_range = kwargs.get("point_cloud_range")

        # Encoder: stem + 4 stages
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D
            ),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True),
        )
        self.stage1 = _build_encoder_stage(cs[0], cs[1], self.D)
        self.stage2 = _build_encoder_stage(cs[1], cs[2], self.D)
        self.stage3 = _build_encoder_stage(cs[2], cs[3], self.D)
        self.stage4 = _build_encoder_stage(cs[3], cs[4], self.D)

        # Decoder: 4 upsampling stages with skip connections
        self.up1 = _build_decoder_stage(cs[4], cs[3], cs[5], self.D)
        self.up2 = _build_decoder_stage(cs[5], cs[2], cs[6], self.D)
        self.up3 = _build_decoder_stage(cs[6], cs[1], cs[7], self.D)
        self.up4 = _build_decoder_stage(cs[7], cs[0], cs[8], self.D)

        _initialize_batchnorm_weights(self)
        self.dropout = nn.Dropout(0.3, True)

    def forward(self, batch_dict: dict[str, Any]) -> dict[str, Any]:
        """Run U-Net encoder-decoder on voxelized point cloud.

        Args:
            batch_dict: Must contain 'voxel_features' (N, C) and 'voxel_coords' (N, 4)
                where coords are [batch_idx, z, y, x].

        Returns:
            batch_dict with added 'point_features' (N, 96) and 'point_coords' (N, 4).
        """
        x = ME.SparseTensor(
            features=batch_dict["voxel_features"],
            coordinates=batch_dict["voxel_coords"].int(),
        )

        # Encoder
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # Decoder with skip connections
        y1 = _decode_stage(self.up1, x4, x3)
        y2 = _decode_stage(self.up2, y1, x2)
        y3 = _decode_stage(self.up3, y2, x1)
        y4 = _decode_stage(self.up4, y3, x0)

        batch_dict["point_features"] = y4.F
        point_coords = common_utils.get_voxel_centers(
            y4.C[:, 1:],
            downsample_times=1,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
        )
        batch_dict["point_coords"] = torch.cat(
            (y4.C[:, 0:1].float(), point_coords), dim=1
        )

        return batch_dict


class MinkUNetSMLP(nn.Module):
    """MinkUNet with SMLP projection head for self-supervised pretraining.

    This variant takes a raw ``ME.SparseTensor`` (NOT batch_dict) and returns
    the decoded sparse tensor. Only used during TARL pretraining, not in the
    OpenPCDet detection pipeline.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        cr: float = kwargs.get("cr", 1.0)
        in_channels: int = kwargs.get("in_channels", 3)
        cs = [int(cr * x) for x in [32, 32, 64, 128, 256, 256, 128, 96, 96]]
        self.run_up: bool = kwargs.get("run_up", True)
        self.D: int = kwargs.get("D", 3)

        # Encoder
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D
            ),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True),
        )
        self.stage1 = _build_encoder_stage(cs[0], cs[1], self.D)
        self.stage2 = _build_encoder_stage(cs[1], cs[2], self.D)
        self.stage3 = _build_encoder_stage(cs[2], cs[3], self.D)
        self.stage4 = _build_encoder_stage(cs[3], cs[4], self.D)

        # Decoder
        self.up1 = _build_decoder_stage(cs[4], cs[3], cs[5], self.D)
        self.up2 = _build_decoder_stage(cs[5], cs[2], cs[6], self.D)
        self.up3 = _build_decoder_stage(cs[6], cs[1], cs[7], self.D)
        self.up4 = _build_decoder_stage(cs[7], cs[0], cs[8], self.D)

        _initialize_batchnorm_weights(self)
        self.dropout = nn.Dropout(0.3, True)
        self.head = SMLP([128, 128, 128])

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """Run U-Net on sparse tensor (pretraining interface).

        Args:
            x: Input sparse tensor with features and coordinates.

        Returns:
            Decoded sparse tensor at input resolution.
        """
        # Encoder
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # Decoder with skip connections
        y1 = _decode_stage(self.up1, x4, x3)
        y2 = _decode_stage(self.up2, y1, x2)
        y3 = _decode_stage(self.up3, y2, x1)
        y4 = _decode_stage(self.up4, y3, x0)

        return y4
