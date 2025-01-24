# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import scipy.signal
import torch
from torch_utils import persistence
from torch_utils import misc
from torch_utils.ops import upfirdn3d
from torch_utils.ops import grid_sample_gradfix
from torch_utils.ops import conv3d_gradfix

#----------------------------------------------------------------------------
# Coefficients of various wavelet decomposition low-pass filters.

wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1':  [0.7071067811865476, 0.7071067811865476],
    'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}

#----------------------------------------------------------------------------
# Helpers for constructing transformation matrices.

def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return misc.constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))

def translate2d(tx, ty, **kwargs):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
        **kwargs)

def translate3d(tx, ty, tz, **kwargs):
    return matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
        **kwargs)

def translate4d(tx, ty, tz, tw, **kwargs):
    return torch.tensor(
        [[1, 0, 0, 0, tx],
         [0, 1, 0, 0, ty],
         [0, 0, 1, 0, tz],
         [0, 0, 0, 1, tw],
         [0, 0, 0, 0, 1]],
        **kwargs
    )

def scale2d(sx, sy, **kwargs):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
        **kwargs)

def scale3d(sx, sy, sz, **kwargs):
    return matrix(
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1],
        **kwargs)


def scale4d(sx, sy, sz, sw, **kwargs):
    return matrix(
        [sx, 0,  0,  0, 0],
        [0,  sy, 0,  0, 0],
        [0,  0,  sz, 0, 0],
        [0,  0,  0,  sw, 1],
        **kwargs)


def rotate2d(theta, **kwargs):
    return matrix(
        [torch.cos(theta), torch.sin(-theta), 0],
        [torch.sin(theta), torch.cos(theta),  0],
        [0,                0,                 1],
        **kwargs)

def rotate3d(v, theta, **kwargs):
    vx = v[0]; vy = v[1]; vz = v[2]
    s = torch.sin(theta); c = torch.cos(theta); cc = 1 - c
    return matrix(
        [vx*vx*cc+c,    vx*vy*cc-vz*s, vx*vz*cc+vy*s, 0],
        [vy*vx*cc+vz*s, vy*vy*cc+c,    vy*vz*cc-vx*s, 0],
        [vz*vx*cc-vy*s, vz*vy*cc+vx*s, vz*vz*cc+c,    0],
        [0,             0,             0,             1],
        **kwargs)

def rotate4d(v, theta, **kwargs):
    assert v.shape[-1] == 4, "Rotation vector must be in 4D space"
    
    vx, vy, vz, vw = v[0], v[1], v[2], v[3]
    s = torch.sin(theta)
    c = torch.cos(theta)
    cc = 1 - c
    
    return torch.tensor([
        [vx * vx * cc + c,    vx * vy * cc - vw * s, vx * vz * cc + vy * s, vx * vw * cc - vz * s, 0],
        [vy * vx * cc + vw * s, vy * vy * cc + c,    vy * vz * cc - vx * s, vy * vw * cc - vz * s, 0],
        [vz * vx * cc - vy * s, vz * vy * cc + vx * s, vz * vz * cc + c,    vz * vw * cc + vy * s, 0],
        [vw * vx * cc + vz * s, vw * vy * cc - vx * s, vw * vz * cc - vy * s, vw * vw * cc + c,    0],
        [0,                    0,                    0,                    0,                    1]
    ], **kwargs)


def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)

def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)

def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)

def translate3d_inv(tx, ty, tz, **kwargs):
    return translate3d(-tx, -ty, -tz, **kwargs)

def scale3d_inv(sx, sy, sz, **kwargs):
    return scale3d(1 / sx, 1 / sy, 1 / sz, **kwargs)

def rotate3d_inv(v, theta, **kwargs):
    # Negate the angle to invert the rotation
    return rotate3d(v, -theta, **kwargs)


#----------------------------------------------------------------------------
# Versatile image augmentation pipeline from the paper
# "Training Generative Adversarial Networks with Limited Data".
#
# All augmentations are disabled by default; individual augmentations can
# be enabled by setting their probability multipliers to 1.

@persistence.persistent_class
class AugmentPipe(torch.nn.Module):
    def __init__(self,
        xflip=0, rotate90=0, xint=0, xint_max=0.125,
        scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=1, aniso_std=0.2, xfrac_std=0.125,
        brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=1,
        imgfilter=0, imgfilter_bands=[1,1,1,1], imgfilter_std=1,
        noise=0, cutout=0, noise_std=0.1, cutout_size=0.5,
    ):
        super().__init__()
        self.register_buffer('p', torch.ones([]))       # Overall multiplier for augmentation probability.

        # Pixel blitting.
        self.xflip            = float(xflip)            # Probability multiplier for x-flip.
        self.rotate90         = float(rotate90)         # Probability multiplier for 90 degree rotations.
        self.xint             = float(xint)             # Probability multiplier for integer translation.
        self.xint_max         = float(xint_max)         # Range of integer translation, relative to image dimensions.

        # General geometric transformations.
        self.scale            = float(scale)            # Probability multiplier for isotropic scaling.
        self.rotate           = float(rotate)           # Probability multiplier for arbitrary rotation.
        self.aniso            = float(aniso)            # Probability multiplier for anisotropic scaling.
        self.xfrac            = float(xfrac)            # Probability multiplier for fractional translation.
        self.scale_std        = float(scale_std)        # Log2 standard deviation of isotropic scaling.
        self.rotate_max       = float(rotate_max)       # Range of arbitrary rotation, 1 = full circle.
        self.aniso_std        = float(aniso_std)        # Log2 standard deviation of anisotropic scaling.
        self.xfrac_std        = float(xfrac_std)        # Standard deviation of frational translation, relative to image dimensions.

        # Color transformations.
        self.brightness       = float(brightness)       # Probability multiplier for brightness.
        self.contrast         = float(contrast)         # Probability multiplier for contrast.
        self.lumaflip         = float(lumaflip)         # Probability multiplier for luma flip.
        self.hue              = float(hue)              # Probability multiplier for hue rotation.
        self.saturation       = float(saturation)       # Probability multiplier for saturation.
        self.brightness_std   = float(brightness_std)   # Standard deviation of brightness.
        self.contrast_std     = float(contrast_std)     # Log2 standard deviation of contrast.
        self.hue_max          = float(hue_max)          # Range of hue rotation, 1 = full circle.
        self.saturation_std   = float(saturation_std)   # Log2 standard deviation of saturation.

        # Image-space filtering.
        self.imgfilter        = float(imgfilter)        # Probability multiplier for image-space filtering.
        self.imgfilter_bands  = list(imgfilter_bands)   # Probability multipliers for individual frequency bands.
        self.imgfilter_std    = float(imgfilter_std)    # Log2 standard deviation of image-space filter amplification.

        # Image-space corruptions.
        self.noise            = float(noise)            # Probability multiplier for additive RGB noise.
        self.cutout           = float(cutout)           # Probability multiplier for cutout.
        self.noise_std        = float(noise_std)        # Standard deviation of additive RGB noise.
        self.cutout_size      = float(cutout_size)      # Size of the cutout rectangle, relative to image dimensions.

        # Setup orthogonal lowpass filter for geometric augmentations.
        self.register_buffer('Hz_geom', upfirdn3d.setup_filter(wavelets['sym6'], separable=False))

        # Construct filter bank for image-space filtering.
        Hz_lo = np.asarray(wavelets['sym2'])            # H(z)
        Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size)) # H(-z)
        Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2    # H(z) * H(z^-1) / 2
        Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2    # H(-z) * H(-z^-1) / 2
        Hz_fbank = np.eye(4, 1)                         # Bandpass(H(z), b_i)
        for i in range(1, Hz_fbank.shape[0]):
            Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
            Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
            Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2 : (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
        self.register_buffer('Hz_fbank', torch.as_tensor(Hz_fbank, dtype=torch.float32))


    def forward(self, images, debug_percentile=None):
        assert isinstance(images, torch.Tensor) and images.ndim == 5
        batch_size, num_channels, depth, height, width = images.shape
        assert num_channels in [1, 3]
        
        device = images.device
        if debug_percentile is not None:
            debug_percentile = torch.as_tensor(debug_percentile, dtype=torch.float32, device=device)

        # -------------------------------------
        # Select parameters for pixel blitting.
        # -------------------------------------

        # # Initialize inverse homogeneous 2D transform: G_inv @ pixel_out ==> pixel_in
        I_4 = torch.eye(4, device=device)
        G_inv = I_4

        # # Apply x-flip with probability (xflip * strength).
        if self.xflip > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 2)  # Randomly select flip (0 or 1)
            i = torch.where(torch.rand([batch_size], device=device) < 1, i, torch.zeros_like(i))
            G_inv = G_inv @ scale3d_inv(1 - 2 * i, 1, 1)

        # # Apply 90 degree rotations with probability (rotate90 * strength).
        if self.rotate90 > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 4)  # Randomly pick 0, 1, 2, or 3
            i = torch.where(torch.rand([batch_size], device=device) < self.rotate90 * self.p, i, torch.zeros_like(i))
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 4))
            
            G_inv = G_inv @ rotate3d([0, 0, 1], -np.pi / 2 * i)

        if self.xint > 0:
            # Generate random translations for x, y, and z within the range [-xint_max, xint_max]
            t = (torch.rand([batch_size, 3], device=device) * 2 - 1) * self.xint_max
            
            # Apply translations based on the probability `self.xint * self.p`
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xint * self.p, t, torch.zeros_like(t))
            
            # Debug mode: Use fixed percentile value for translation
            if debug_percentile is not None:
                t = torch.full_like(t, (debug_percentile * 2 - 1) * self.xint_max)
            
            # Apply integer translations (rounded) to G_inv
            G_inv = G_inv @ translate3d_inv(
                torch.round(t[:, 0] * width),   # Translate in x-direction
                torch.round(t[:, 1] * height), # Translate in y-direction
                torch.round(t[:, 2] * depth)   # Translate in z-direction
            )


        # # --------------------------------------------------------
        # # Select parameters for general geometric transformations.
        # # --------------------------------------------------------

        # # Apply isotropic scaling with probability (scale * strength).
        if self.scale > 0:
            # Generate random scaling factors (log2-normal distributed)
            s = torch.exp2(torch.randn([batch_size], device=device) * self.scale_std)
            
            # Apply scaling based on probability
            s = torch.where(torch.rand([batch_size], device=device) < self.scale * self.p, s, torch.ones_like(s))
            
            # Debug mode: Use fixed percentile for scaling
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.scale_std))
            
            # Apply uniform scaling in all 3 dimensions
            G_inv = G_inv @ scale3d_inv(s, s, s)


        # # Apply pre-rotation with probability p_rot.
        p_rot = 1 - torch.sqrt((1 - self.rotate * self.p).clamp(0, 1))  # Probability of applying prerotation
        if self.rotate > 0:
            # Generate random rotation angles in radians
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            
            # Apply rotations based on probability
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            
            # Debug mode: Use fixed percentile for rotation angle
            if debug_percentile is not None:
                theta = torch.full_like(theta, (debug_percentile * 2 - 1) * np.pi * self.rotate_max)
            
            # Apply prerotation (about the z-axis, for simplicity)
            G_inv = G_inv @ rotate3d([0, 0, 1], -theta)

        # # Apply anisotropic scaling with probability (aniso * strength).
        if self.aniso > 0:
            # Generate random anisotropic scaling factors (log2-normal distributed)
            s = torch.exp2(torch.randn([batch_size], device=device) * self.aniso_std)
            
            # Apply anisotropic scaling based on probability
            s = torch.where(torch.rand([batch_size], device=device) < self.aniso * self.p, s, torch.ones_like(s))
            
            # Debug mode: Use fixed anisotropic scaling
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.aniso_std))
            
            # Apply anisotropic scaling to x and y, leaving z unchanged
            G_inv = G_inv @ scale3d_inv(s, 1 / s, 1)


        # # Apply post-rotation with probability p_rot.
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            if debug_percentile is not None:
                theta = torch.zeros_like(theta)
            
            G_inv = G_inv @ rotate3d([0, 0, 1], -theta)

        # # Apply fractional translation with probability (xfrac * strength).
        if self.xfrac > 0:
            # Generate random fractional displacements for x, y, and z
            t = torch.randn([batch_size, 3], device=device) * self.xfrac_std

            # Apply fractional translations based on probability
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xfrac * self.p, t, torch.zeros_like(t))

            # Debug mode: Use fixed fractional displacement
            if debug_percentile is not None:
                t = torch.full_like(t, torch.erfinv(debug_percentile * 2 - 1) * self.xfrac_std)

            # Apply the translation (scaled by respective dimensions)
            G_inv = G_inv @ translate3d_inv(t[:, 0] * width, t[:, 1] * height, t[:, 2] * depth)

        # # ----------------------------------
        # # Execute geometric transformations.
        # # ----------------------------------

        if G_inv is not I_4:

            # Calculate padding.
            cx = (width - 1) / 2
            cy = (height - 1) / 2
            cz = (depth - 1) / 2
            cp = torch.tensor([
            [-cx, -cy, -cz, 1],
            [ cx, -cy, -cz, 1],
            [ cx,  cy, -cz, 1],
            [-cx,  cy, -cz, 1],
            [-cx, -cy,  cz, 1],
            [ cx, -cy,  cz, 1],
            [ cx,  cy,  cz, 1],
            [-cx,  cy,  cz, 1],
            ], device=device)
            cp = G_inv @ cp.t()

            Hz_pad = self.Hz_geom.shape[0] // 4
            margin = cp[:, :3, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]

            margin = torch.cat([-margin, margin]).max(dim=1).values
            margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy, Hz_pad * 2 - cz] * 2, device=device)
            margin = margin.max(misc.constant([0, 0, 0] * 2, device=device))
            margin = margin.min(misc.constant([width-1, height-1, depth-1] * 2, device=device))
            mx0, my0, mz0, mx1, my1, mz1 = margin.ceil().to(torch.int32)
            # tensor = torch.permute(tensor, (0, 4, 1, 2, 3)) # Don't know if needed :thinking:
            tensor = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1,mz0,mz1], mode='reflect')
            G_inv = translate3d((mx0 - mx1) / 2, (my0 - my1) / 2, (mz0 - mz1) / 2) @ G_inv

            # Upsample.
            images = upfirdn3d.upsample3d(x=tensor, f=self.Hz_geom, up=2)
            G_inv = scale3d(2, 2, 2, device=device) @ G_inv @ scale3d_inv(2, 2, 2, device=device)
            G_inv = translate3d(-0.5, -0.5, -0.5, device=device) @ G_inv @ translate3d_inv(-0.5, -0.5, -0.5, device=device)
            shape = [batch_size, num_channels, (height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2, (depth + Hz_pad * 2) * 2]
            G_inv = scale3d(2 / images.shape[4], 2 / images.shape[3], 2 / images.shape[2], device=device) @ G_inv @ scale3d_inv(2 / images.shape[4], 2 / shape[3], 2 / shape[2], device=device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:, :3, :], size=shape, align_corners=False)
            images = grid_sample_gradfix.grid_sample(images, grid)
            images = upfirdn3d.downsample3d(x=images, f=self.Hz_geom, down=2, padding=-Hz_pad*2, flip_filter=True)

        # --------------------------------------------
        # Select parameters for color transformations.
        # --------------------------------------------

        # Initialize homogeneous 3D transformation matrix: C @ color_in ==> color_out
        I_4 = torch.eye(4, device=device)
        C = I_4

        # Apply brightness with probability (brightness * strength).
        # 3D READY
        if self.brightness > 0:
            b = torch.randn([batch_size], device=device) * self.brightness_std
            b = torch.where(torch.rand([batch_size], device=device) < self.brightness * self.p, b, torch.zeros_like(b))
            if debug_percentile is not None:
                b = torch.full_like(b, torch.erfinv(debug_percentile * 2 - 1) * self.brightness_std)
            C = translate3d(b, b, b) @ C

        # Apply contrast with probability (contrast * strength).
        # 3D READY
        if self.contrast > 0:
            c = torch.exp2(torch.randn([batch_size], device=device) * self.contrast_std)
            c = torch.where(torch.rand([batch_size], device=device) < self.contrast * self.p, c, torch.ones_like(c))
            if debug_percentile is not None:
                c = torch.full_like(c, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.contrast_std))
            C = scale3d(c, c, c) @ C

        # Apply luma flip with probability (lumaflip * strength).
        # 3D READY
        v = misc.constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=device) # Luma axis.
        if self.lumaflip > 0:
            i = torch.floor(torch.rand([batch_size, 1, 1], device=device) * 2)
            i = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.lumaflip * self.p, i, torch.zeros_like(i))
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 2))
            C = (I_4 - 2 * v.ger(v) * i) @ C # Householder reflection.

        # Apply hue rotation with probability (hue * strength).
        # 3D READY
        if self.hue > 0 and num_channels > 1:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.hue_max
            theta = torch.where(torch.rand([batch_size], device=device) < self.hue * self.p, theta, torch.zeros_like(theta))
            if debug_percentile is not None:
                theta = torch.full_like(theta, (debug_percentile * 2 - 1) * np.pi * self.hue_max)
            C = rotate3d(v, theta) @ C # Rotate around v.

        # Apply saturation with probability (saturation * strength).
        # 3D READY
        if self.saturation > 0 and num_channels > 1:
            s = torch.exp2(torch.randn([batch_size, 1, 1], device=device) * self.saturation_std)
            s = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.saturation * self.p, s, torch.ones_like(s))
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.saturation_std))
            C = (v.ger(v) + (I_4 - v.ger(v)) * s) @ C

        # ------------------------------
        # Execute color transformations.
        # ------------------------------

        # # Execute if the transform is not identity.
        # 3D READY
        if C is not I_4:
            images = images.reshape([batch_size, num_channels, depth * height * width])  # Adjusted for 3D
            if num_channels == 3:
                images = images.permute(0, 2, 1)  # (batch_size, N, 3)
                images_hom = torch.cat([images, torch.ones(batch_size, depth * height * width, 1, device=device)], dim=2)  # (batch_size, N, 4)
                transformed = torch.bmm(images_hom, C.transpose(1, 2))  # (batch_size, N, 4)
                images = transformed[:, :, :3].permute(0, 2, 1)  # (batch_size, 3, N)
            elif num_channels == 1:
                C_mean = C[:, :3, :].mean(dim=1, keepdims=True)  # (batch_size, 1, 4)
                C_scale = C_mean[:, :, :3].sum(dim=2, keepdims=True)  # (batch_size, 1, 1)
                C_offset = C_mean[:, :, 3:]  # (batch_size, 1, 1)
                images = images * C_scale + C_offset  # (batch_size, 1, N)
            else:
                raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
            images = images.reshape([batch_size, num_channels, depth, height, width])

        # # ----------------------
        # # Image-space filtering.
        # # ----------------------
        # 3D READY
        if self.imgfilter > 0:
            num_bands = self.Hz_fbank.shape[0]
            assert len(self.imgfilter_bands) == num_bands
            expected_power = misc.constant(np.array([10, 1, 1, 1]) / 13, device=device)  # Expected power spectrum (1/f).

            # Apply amplification for each band with probability (imgfilter * strength * band_strength).
            g = torch.ones([batch_size, num_bands], device=device)  # Global gain vector (identity).
            for i, band_strength in enumerate(self.imgfilter_bands):
                t_i = torch.exp2(torch.randn([batch_size], device=device) * self.imgfilter_std)
                t_i = torch.where(torch.rand([batch_size], device=device) < self.imgfilter * self.p * band_strength, t_i, torch.ones_like(t_i))
                if debug_percentile is not None:
                    t_i = torch.full_like(t_i, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.imgfilter_std)) if band_strength > 0 else torch.ones_like(t_i)
                t = torch.ones([batch_size, num_bands], device=device)                   # Temporary gain vector.
                t[:, i] = t_i                                                            # Replace i'th element.
                t = t / (expected_power * t.square()).sum(dim=-1, keepdims=True).sqrt()  # Normalize power.
                g = g * t                                                                # Accumulate into global gain.

            # Construct combined amplification filter.
            Hz_prime = g @ self.Hz_fbank                                      # [batch, tap]
            Hz_prime = Hz_prime.unsqueeze(1).repeat(1, num_channels, 1)       # [batch, channels, tap]
            Hz_prime = Hz_prime.reshape(batch_size * num_channels, 1, -1)     # [batch * channels, 1, tap]

            # Apply filter.
            p = self.Hz_fbank.shape[1] // 2
            images = images.reshape(1, batch_size * num_channels, depth, height, width)
            images = torch.nn.functional.pad(images, [p, p, p, p, p, p], mode='reflect')  # Pad depth, height, width.

            # Prepare filters for each axis.
            weight_d = Hz_prime.view(batch_size * num_channels, 1, -1, 1, 1)  # Depth filter.
            weight_h = Hz_prime.view(batch_size * num_channels, 1, 1, -1, 1)  # Height filter.
            weight_w = Hz_prime.view(batch_size * num_channels, 1, 1, 1, -1)  # Width filter.

            # Apply convolutions along each axis.
            images = conv3d_gradfix.conv3d(images, weight=weight_d, groups=batch_size * num_channels)
            images = conv3d_gradfix.conv3d(images, weight=weight_h, groups=batch_size * num_channels)
            images = conv3d_gradfix.conv3d(images, weight=weight_w, groups=batch_size * num_channels)

            images = images.reshape(batch_size, num_channels, depth, height, width)


        # ------------------------
        # Image-space corruptions.
        # ------------------------

        # Apply additive RGB noise with probability (noise * strength).
        # 3D READY
        if self.noise > 0:
            sigma = torch.randn([batch_size, 1, 1, 1, 1], device=device).abs() * self.noise_std
            sigma = torch.where(torch.rand([batch_size, 1, 1, 1, 1], device=device) < self.noise * self.p, sigma, torch.zeros_like(sigma))
            if debug_percentile is not None:
                sigma = torch.full_like(sigma, torch.erfinv(debug_percentile) * self.noise_std)
            images = images + torch.randn([batch_size, num_channels, depth, height, width], device=device) * sigma

        # Apply cutout with probability (cutout * strength).
        # 3D READY
        if self.cutout > 0:
            size = torch.full([batch_size, 3, 1, 1, 1, 1], self.cutout_size, device=device)
            size = torch.where(
                torch.rand([batch_size, 1, 1, 1, 1, 1], device=device) < self.cutout * self.p, 
                size, 
                torch.zeros_like(size)
            )
            center = torch.rand([batch_size, 3, 1, 1, 1, 1], device=device)
            if debug_percentile is not None:
                size = torch.full_like(size, self.cutout_size)
                center = torch.full_like(center, debug_percentile)
            
            # Create coordinates for depth, height, and width
            coord_z = torch.arange(depth, device=device).reshape([1, 1, -1, 1, 1])
            coord_y = torch.arange(height, device=device).reshape([1, 1, 1, -1, 1])
            coord_x = torch.arange(width, device=device).reshape([1, 1, 1, 1, -1])

            # Compute the masks for each dimension
            mask_z = (((coord_z + 0.5) / depth - center[:, 0]).abs() >= size[:, 0] / 2)
            mask_y = (((coord_y + 0.5) / height - center[:, 1]).abs() >= size[:, 1] / 2)
            mask_x = (((coord_x + 0.5) / width - center[:, 2]).abs() >= size[:, 2] / 2)

            # Combine masks to create the final 3D mask
            mask = torch.logical_or(torch.logical_or(mask_z, mask_y), mask_x).to(torch.float32)
            
            # Apply the mask to the volumetric data
            images = images * mask


        return images

#----------------------------------------------------------------------------
