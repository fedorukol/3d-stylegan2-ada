import os
import warnings
import numpy as np
import torch
import traceback

from .. import custom_ops
from .. import misc
from . import conv3d_gradfix 

#----------------------------------------------------------------------------

_inited = False
_plugin = None

def _init():
    global _inited, _plugin
    if not _inited:
        sources = ['upfirdn3d.cpp', 'upfirdn3d.cu']
        sources = [os.path.join(os.path.dirname(__file__), s) for s in sources]
        try:
            _plugin = custom_ops.get_plugin('upfirdn3d_plugin', sources=sources, extra_cuda_cflags=['--use_fast_math'])
        except:
            warnings.warn('Failed to build CUDA kernels for upfirdn3d. Falling back to slow reference implementation. Details:\n\n' + traceback.format_exc())
    return _plugin is not None

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy, sz = scaling
    assert sx >= 1 and sy >= 1 and sz >= 1
    return sx, sy, sz

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 3:
        padx, pady, padz = padding
        padding = [padx, padx, pady, pady, padz, padz]
    padx0, padx1, pady0, pady1, padz0, padz1 = padding
    return padx0, padx1, pady0, pady1, padz0, padz1

def _get_filter_size(f):
    if f is None:
        return 1, 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2, 3]
    fd = f.shape[0]
    fh = f.shape[1] if f.ndim > 1 else 1
    fw = f.shape[2] if f.ndim > 2 else 1
    with misc.suppress_tracer_warnings():
        fw = int(fw)
        fh = int(fh)
        fd = int(fd)
    misc.assert_shape(f, [fd, fh, fw][:f.ndim])
    assert fw >= 1 and fh >= 1 and fd >= 1
    return fw, fh, fd

#----------------------------------------------------------------------------

def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    r"""Convenience function to setup 3D FIR filter for `upfirdn3d()`.

    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_depth, filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_depth, filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2, 3]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = torch.einsum('i,j,k->ijk', f, f, f)
    elif f.ndim == 2 and not separable:
        f = f.unsqueeze(0).repeat(f.shape[0], 1, 1)
    assert f.ndim == (1 if separable else 3)

    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f

#----------------------------------------------------------------------------

def upfirdn3d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Pad, upsample, filter, and downsample a batch of 3D volumes.

    Performs the following sequence of operations for each channel:

    1. Upsample the volume by inserting N-1 zeros after each voxel (`up`).

    2. Pad the volume with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the volume.

    3. Convolve the volume with the specified 3D FIR filter (`f`), shrinking it
       so that the footprint of all output voxels lies within the input volume.

    4. Downsample the volume by keeping every Nth voxel (`down`).

    This sequence of operations is a 3D extension of the 2D upfirdn.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_depth, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_depth, filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y, z]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y, z]` (default: 1).
        padding:     Padding with respect to the upsampled volume. Can be a single number
                     or a list/tuple `[x, y, z]` or `[x_before, x_after, y_before, y_after, z_before, z_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_depth, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    return _upfirdn3d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)

#----------------------------------------------------------------------------

@misc.profiled_function
def _upfirdn3d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn3d()` using standard PyTorch ops.
    """
    assert isinstance(x, torch.Tensor) and x.ndim == 5
    if f is None:
        f = torch.ones([1, 1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 3]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_depth, in_height, in_width = x.shape
    upx, upy, upz = _parse_scaling(up)
    downx, downy, downz = _parse_scaling(down)
    padx0, padx1, pady0, pady1, padz0, padz1 = _parse_padding(padding)

    x = x.reshape([batch_size, num_channels, in_depth, 1, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1, 0, 0, 0, upz - 1])
    x = x.reshape([batch_size, num_channels, in_depth * upz, in_height * upy, in_width * upx])

    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0),
                                    max(pady0, 0), max(pady1, 0),
                                    max(padz0, 0), max(padz1, 0)])
    x = x[:, :, max(-padz0, 0): x.shape[2] - max(-padz1, 0),
             max(-pady0, 0): x.shape[3] - max(-pady1, 0),
             max(-padx0, 0): x.shape[4] - max(-padx1, 0)]

    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    if f.ndim == 1:
        f_d = f.view(1, 1, -1, 1, 1)
        f_h = f.view(1, 1, 1, -1, 1)
        f_w = f.view(1, 1, 1, 1, -1)

        x = conv3d_gradfix.conv3d(input=x, weight=f_d, groups=num_channels)
        x = conv3d_gradfix.conv3d(input=x, weight=f_h, groups=num_channels)
        x = conv3d_gradfix.conv3d(input=x, weight=f_w, groups=num_channels)
    elif f.ndim == 3:
        f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
        x = conv3d_gradfix.conv3d(input=x, weight=f, groups=num_channels)
    else:
        raise ValueError('Unsupported filter dimensions.')

    x = x[:, :, ::downz, ::downy, ::downx]
    return x

#----------------------------------------------------------------------------

def filter3d(x, f, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Filter a batch of 3D volumes using the given 3D FIR filter.

    By default, the result is padded so that its shape matches the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Voxels outside the volume are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_depth, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_depth, filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y, z]` or `[x_before, x_after, y_before, y_after, z_before, z_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_depth, out_height, out_width]`.
    """
    padx0, padx1, pady0, pady1, padz0, padz1 = _parse_padding(padding)
    fw, fh, fd = _get_filter_size(f)
    p = [
        padx0 + fw // 2,
        padx1 + (fw - 1) // 2,
        pady0 + fh // 2,
        pady1 + (fh - 1) // 2,
        padz0 + fd // 2,
        padz1 + (fd - 1) // 2,
    ]
    return upfirdn3d(x, f, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

def upsample3d(x, f, up=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Upsample a batch of 3D volumes using the given 3D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Voxels outside the volume are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_depth, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_depth, filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y, z]` (default: 1).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y, z]` or `[x_before, x_after, y_before, y_after, z_before, z_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_depth, out_height, out_width]`.
    """
    upx, upy, upz = _parse_scaling(up)
    padx0, padx1, pady0, pady1, padz0, padz1 = _parse_padding(padding)
    fw, fh, fd = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
        padz0 + (fd + upz - 1) // 2,
        padz1 + (fd - upz) // 2,
    ]
    return upfirdn3d(x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain*upx*upy*upz, impl=impl)

def downsample3d(x, f, down=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Downsample a batch of 3D volumes using the given 3D FIR filter.

    By default, the result is padded so that its shape is a fraction of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Voxels outside the volume are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_depth, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_depth, filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y, z]` (default: 1).
        padding:     Padding with respect to the input. Can be a single number or a
                     list/tuple `[x, y, z]` or `[x_before, x_after, y_before, y_after, z_before, z_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_depth, out_height, out_width]`.
    """
    downx, downy, downz = _parse_scaling(down)
    padx0, padx1, pady0, pady1, padz0, padz1 = _parse_padding(padding)
    fw, fh, fd = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
        padz0 + (fd - downz + 1) // 2,
        padz1 + (fd - downz) // 2,
    ]
    return upfirdn3d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)
