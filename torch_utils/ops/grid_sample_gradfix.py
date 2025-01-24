# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom replacement for `torch.nn.functional.grid_sample` that
supports arbitrarily high order gradients between the input and output.
Only works on 2D images and assumes
`mode='bilinear'`, `padding_mode='zeros'`, `align_corners=False`."""

import warnings
import torch

# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

#----------------------------------------------------------------------------

enabled = False  # Enable the custom op by setting this to true.

#----------------------------------------------------------------------------

def grid_sample(image, optical):
    N, C, D, H, W = image.shape
    _, D_out, H_out, W_out, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (W - 1)
    iy = ((iy + 1) / 2) * (H - 1)
    iz = ((iz + 1) / 2) * (D - 1)

    with torch.no_grad():
        ix_0 = torch.floor(ix)
        iy_0 = torch.floor(iy)
        iz_0 = torch.floor(iz)

        ix_1 = ix_0 + 1
        iy_1 = iy_0 + 1
        iz_1 = iz_0 + 1

    dx = ix - ix_0
    dy = iy - iy_0
    dz = iz - iz_0

    w000 = (1 - dx) * (1 - dy) * (1 - dz)
    w100 = dx        * (1 - dy) * (1 - dz)
    w010 = (1 - dx) * dy        * (1 - dz)
    w110 = dx        * dy        * (1 - dz)
    w001 = (1 - dx) * (1 - dy) * dz
    w101 = dx        * (1 - dy) * dz
    w011 = (1 - dx) * dy        * dz
    w111 = dx        * dy        * dz

    with torch.no_grad():
        ix_0 = torch.clamp(ix_0, 0, W - 1)
        ix_1 = torch.clamp(ix_1, 0, W - 1)
        iy_0 = torch.clamp(iy_0, 0, H - 1)
        iy_1 = torch.clamp(iy_1, 0, H - 1)
        iz_0 = torch.clamp(iz_0, 0, D - 1)
        iz_1 = torch.clamp(iz_1, 0, D - 1)

    image_flat = image.view(N, C, D * H * W)

    def gather_vals(ix_idx, iy_idx, iz_idx):
        linear_idx = (iz_idx * H * W) + (iy_idx * W) + ix_idx
        linear_idx = linear_idx.long()
        idx_expanded = linear_idx.view(N, 1, D_out * H_out * W_out).expand(N, C, D_out * H_out * W_out)
        val = torch.gather(image_flat, 2, idx_expanded)
        return val.view(N, C, D_out, H_out, W_out)

    val000 = gather_vals(ix_0, iy_0, iz_0)
    val100 = gather_vals(ix_1, iy_0, iz_0)
    val010 = gather_vals(ix_0, iy_1, iz_0)
    val110 = gather_vals(ix_1, iy_1, iz_0)
    val001 = gather_vals(ix_0, iy_0, iz_1)
    val101 = gather_vals(ix_1, iy_0, iz_1)
    val011 = gather_vals(ix_0, iy_1, iz_1)
    val111 = gather_vals(ix_1, iy_1, iz_1)

    out_val = (
        val000 * w000.unsqueeze(1) +
        val100 * w100.unsqueeze(1) +
        val010 * w010.unsqueeze(1) +
        val110 * w110.unsqueeze(1) +
        val001 * w001.unsqueeze(1) +
        val101 * w101.unsqueeze(1) +
        val011 * w011.unsqueeze(1) +
        val111 * w111.unsqueeze(1)
    )

    return out_val

#----------------------------------------------------------------------------
