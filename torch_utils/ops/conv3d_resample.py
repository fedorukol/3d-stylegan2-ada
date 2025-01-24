"""3D convolution with optional up/downsampling."""

import torch

from .. import misc
from . import conv3d_gradfix
from . import upfirdn3d
from .upfirdn3d import _parse_padding
from .upfirdn3d import _get_filter_size

#----------------------------------------------------------------------------

def _get_weight_shape(w):
    with misc.suppress_tracer_warnings():
        shape = [int(sz) for sz in w.shape]
    misc.assert_shape(w, shape)
    return shape

#----------------------------------------------------------------------------

def _conv3d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):
    """Wrapper for the underlying `conv3d()` and `conv_transpose3d()` implementations.
    """
    out_channels, in_channels_per_group, kd, kh, kw = _get_weight_shape(w)

    if not flip_weight: 
        w = w.flip([2, 3, 4])

    if kw == 1 and kh == 1 and kd == 1 and stride == 1 and padding in [0, [0, 0, 0], (0, 0, 0)] and not transpose:
        if x.stride()[1] == 1 and min(out_channels, in_channels_per_group) < 64:
            if out_channels <= 4 and groups == 1:
                in_shape = x.shape
                x = w.squeeze(4).squeeze(3).squeeze(2) @ x.reshape([in_shape[0], in_channels_per_group, -1])
                x = x.reshape([in_shape[0], out_channels, in_shape[2], in_shape[3], in_shape[4]])
            else:
                x = x.to(memory_format=torch.contiguous_format)
                w = w.to(memory_format=torch.contiguous_format)
                x = conv3d_gradfix.conv3d(x, w, groups=groups)
            return x.to(memory_format=torch.channels_last)

    op = conv3d_gradfix.conv_transpose3d if transpose else conv3d_gradfix.conv3d
    return op(x, w, stride=stride, padding=padding, groups=groups)

#----------------------------------------------------------------------------

@misc.profiled_function
def conv3d_resample(x, w, f=None, up=1, down=1, padding=0, groups=1, flip_weight=True, flip_filter=False):
    r"""3D convolution with optional up/downsampling.

    Padding is performed only once at the beginning, not between the operations.

    Args:
        x:              Input tensor of shape
                        `[batch_size, in_channels, in_depth, in_height, in_width]`.
        w:              Weight tensor of shape
                        `[out_channels, in_channels//groups, kernel_depth, kernel_height, kernel_width]`.
        f:              Low-pass filter for up/downsampling. Must be prepared beforehand by
                        calling upfirdn3d.setup_filter(). None = identity (default).
        up:             Integer upsampling factor (default: 1).
        down:           Integer downsampling factor (default: 1).
        padding:        Padding with respect to the upsampled volume. Can be a single number
                        or a list/tuple `[x, y, z]` or `[x_before, x_after, y_before, y_after, z_before, z_after]`
                        (default: 0).
        groups:         Split input channels into N groups (default: 1).
        flip_weight:    False = convolution, True = correlation (default: True).
        flip_filter:    False = convolution, True = correlation (default: False).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_depth, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor) and (x.ndim == 5)
    assert isinstance(w, torch.Tensor) and (w.ndim == 5) and (w.dtype == x.dtype)
    assert f is None or (isinstance(f, torch.Tensor) and f.ndim in [1, 3] and f.dtype == torch.float32)
    assert isinstance(up, int) and (up >= 1)
    assert isinstance(down, int) and (down >= 1)
    assert isinstance(groups, int) and (groups >= 1)
    out_channels, in_channels_per_group, kd, kh, kw = _get_weight_shape(w)
    fd, fh, fw = _get_filter_size(f)
    px0, px1, py0, py1, pz0, pz1 = _parse_padding(padding)

    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
        pz0 += (fd + up - 1) // 2
        pz1 += (fd - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2
        pz0 += (fd - down + 1) // 2
        pz1 += (fd - down) // 2

    if kw == 1 and kh == 1 and kd == 1 and (down > 1 and up == 1):
        x = upfirdn3d.upfirdn3d(x=x, f=f, down=down, padding=[px0, px1, py0, py1, pz0, pz1], flip_filter=flip_filter)
        x = _conv3d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        return x

    if kw == 1 and kh == 1 and kd == 1 and (up > 1 and down == 1):
        x = _conv3d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        x = upfirdn3d.upfirdn3d(x=x, f=f, up=up, padding=[px0, px1, py0, py1, pz0, pz1], gain=up**3, flip_filter=flip_filter)
        return x

    if down > 1 and up == 1:
        x = upfirdn3d.upfirdn3d(x=x, f=f, padding=[px0, px1, py0, py1, pz0, pz1], flip_filter=flip_filter)
        x = _conv3d_wrapper(x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight)
        return x

    if up > 1:
        if groups == 1:
            w = w.transpose(0, 1)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kd, kh, kw)
            w = w.transpose(1, 2)
            w = w.reshape(groups * in_channels_per_group, out_channels // groups, kd, kh, kw)
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pz0 -= kd - 1
        pz1 -= kd - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        pzt = max(min(-pz0, -pz1), 0)
        x = _conv3d_wrapper(x=x, w=w, stride=up, padding=[pzt, pyt, pxt], groups=groups, transpose=True, flip_weight=(not flip_weight))
        x = upfirdn3d.upfirdn3d(x=x, f=f, padding=[px0+pxt, px1+pxt, py0+pyt, py1+pyt, pz0+pzt, pz1+pzt], gain=up**3, flip_filter=flip_filter)
        if down > 1:
            x = upfirdn3d.upfirdn3d(x=x, f=f, down=down, flip_filter=flip_filter)
        return x

    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and pz0 == pz1 and px0 >= 0 and py0 >= 0 and pz0 >= 0:
            return _conv3d_wrapper(x=x, w=w, padding=[pz0, py0, px0], groups=groups, flip_weight=flip_weight)

    x = upfirdn3d.upfirdn3d(x=x, f=(f if up > 1 else None), up=up, padding=[px0, px1, py0, py1, pz0, pz1], gain=up**3, flip_filter=flip_filter)
    x = _conv3d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = upfirdn3d.upfirdn3d(x=x, f=f, down=down, flip_filter=flip_filter)
    return x
