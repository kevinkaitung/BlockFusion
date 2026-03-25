"""
PyTorch Ray Marcher with:
  - Perspective projection (non-parallel rays per pixel)
  - Precomputed voxel shadow coefficients (no Phong gradient needed)
  - Volume rendering via alpha compositing
  - Trilinear interpolation of volume/shadow data
"""
import warnings
try:
    from pysampler import decode_shadow, decode, create_sampler
except ImportError:
    warnings.warn(
        "Cannot import dvnr sampler. "
        "Only support pytorch implementation for volume scalar values sampling. No shadow coefficients sampling."
    )

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Any
import json5
try:
    from networks import NeurCompNet
except ImportError:
    # HACK: to solve importing from different repos
    from assess_geometry_loss import NeurCompNet
from easydict import EasyDict as edict
import os
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------
# Camera / Scene config
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    """Pinhole camera in world space."""
    position: torch.Tensor        # (3,)
    look_at:  torch.Tensor        # (3,)
    up:       torch.Tensor        # (3,)
    fov_y:    float               # vertical field-of-view in degrees
    width:    int
    height:   int


@dataclass
class MarchConfig:
    t_near:     float = 0.1
    t_far:      float = 10.0
    n_samples:  int   = 128
    density_threshold: float = 1e-4
    patch_width: int = 16
    patch_height: int = 16

# ---------------------------------------------------------------------------
# Transfer Function
# ---------------------------------------------------------------------------
# TODO: tfn function might not need gradient because we just treat it like constant value for lookup
def build_transfer_function(color_controls: list, opacity_controls: list, gaussian_objects: list, lut_size: int = 1024):
    """
    Pre-bakes the transfer function into a lookup table (LUT) of shape (lut_size, 4).
    Supports two opacity formats:
      - opacity_controls: piecewise linear control points {'position': {'x', 'y'}}
      - gaussian_objects: sum of gaussians {'mean', 'sigma', 'heightFactor'}
    One of the two must be provided.

    Args:
        color_controls:   list of {'position': float, 'color': {'r', 'g', 'b'}}
        opacity_controls: list of {'position': {'x': float, 'y': float}}
        gaussian_objects: list of {'mean': float, 'sigma': float, 'heightFactor': float}
        lut_size:         resolution of the baked LUT
    Returns:
        lut: (lut_size, 4) float tensor
    """
    
    assert opacity_controls is not None or gaussian_objects is not None, \
        "Must provide either opacity_controls or gaussian_objects"
    assert not (opacity_controls is not None and gaussian_objects is not None), \
        "Provide only one of opacity_controls or gaussian_objects, not both"
    
    with torch.no_grad():
        # -- sort control points by position --
        color_positions = torch.tensor([c['position']          for c in color_controls])
        color_values    = torch.tensor([[c['color']['r'],
                                        c['color']['g'],
                                        c['color']['b']]      for c in color_controls])  # (N, 3)

        # -- query positions: evenly spaced in [0, 1] --
        t = torch.linspace(0.0, 1.0, lut_size)   # (lut_size,)

        # -- piecewise linear interpolation helper --
        def piecewise_lerp(query, ctrl_pos, ctrl_val):
            """
            query:    (Q,)
            ctrl_pos: (N,)   sorted positions in [0,1]
            ctrl_val: (N, C) or (N,)
            returns:  (Q, C) or (Q,)
            """
            is_1d = ctrl_val.dim() == 1
            if is_1d:
                ctrl_val = ctrl_val.unsqueeze(-1)   # (N, 1)

            Q = query.shape[0]
            C = ctrl_val.shape[-1]
            # out records each position on a line (0~1) should have what color/opacity (map scalar value between 0~1 to corresponding color/opacity)
            out = torch.zeros(Q, C)

            for q in range(Q):
                v = query[q]
                # find the segment [pos_lo, pos_hi] that brackets v
                idx = torch.searchsorted(ctrl_pos, v).clamp(1, len(ctrl_pos) - 1)
                lo, hi = idx - 1, idx
                pos_lo, pos_hi = ctrl_pos[lo], ctrl_pos[hi]
                span = (pos_hi - pos_lo).clamp(min=1e-8)
                alpha = ((v - pos_lo) / span).clamp(0.0, 1.0)
                out[q] = (1.0 - alpha) * ctrl_val[lo] + alpha * ctrl_val[hi]

            return out.squeeze(-1) if is_1d else out
        
        # -- gaussian opacity evaluator --
        def gaussian_opacity(query, gaussians):
            """
            Sum of Gaussians evaluated at each query position.
            f(x) = sum_i [ heightFactor_i * exp(-0.5 * ((x - mean_i) / sigma_i)^2) ]

            query:     (Q,)
            gaussians: list of {'mean', 'sigma', 'heightFactor'}
            returns:   (Q,)  opacity values in [0, 1]
            """
            means   = torch.tensor([g['mean']         for g in gaussians])  # (G,)
            sigmas  = torch.tensor([g['sigma']         for g in gaussians])  # (G,)
            heights = torch.tensor([g['heightFactor']  for g in gaussians])  # (G,)

            # query: (Q,) → (Q, 1),  means/sigmas/heights: (G,) → (1, G)
            q   = query.unsqueeze(-1)                                   # (Q, 1)
            exp = torch.exp(-0.5 * ((q - means) / sigmas.clamp(1e-8)) ** 2)  # (Q, G)
            opacity = (heights * exp).sum(dim=-1)                       # (Q,)

            # clamp to [0, 1] — sum of gaussians can exceed 1 if heights are large
            return opacity.clamp(0.0, 1.0)

        # -- build color LUT --
        lut_rgb   = piecewise_lerp(t, color_positions,   color_values)    # (lut_size, 3)
        
        # -- build opacity LUT --
        if opacity_controls is not None:
            opacity_positions = torch.tensor([o['position']['x'] for o in opacity_controls])
            opacity_values    = torch.tensor([o['position']['y'] for o in opacity_controls])
            lut_alpha = piecewise_lerp(t, opacity_positions, opacity_values)  # (lut_size,)
        else:
            lut_alpha = gaussian_opacity(t, gaussian_objects)                 # (lut_size,)

        lut = torch.cat([lut_rgb, lut_alpha.unsqueeze(-1)], dim=-1)       # (lut_size, 4)
    return lut.detach()     # explicitly detach just to be safe


def sample_transfer_function(lut: torch.Tensor, scalar_values: torch.Tensor):
    """
    Given a pre-baked LUT and raw scalar values in [0, 1], returns RGBA via
    linear interpolation into the LUT.

    Args:
        lut:           (lut_size, 4)  pre-baked transfer function
        scalar_values: (...,)         scalar field values in [0, 1]
    Returns:
        rgba:          (..., 4)       interpolated color and opacity
    """
    lut_size   = lut.shape[0]
    lut        = lut.to(scalar_values.device)
    orig_shape = scalar_values.shape

    # clamp scalars to [0, 1] then map to LUT indices
    sv   = scalar_values.reshape(-1).clamp(0.0, 1.0)     # (N,)
    idx  = sv * (lut_size - 1)                            # directly convert each scalar values into index that can be used in LUT
    lo   = idx.long().clamp(0, lut_size - 2)
    hi   = (lo + 1).clamp(0, lut_size - 1)
    frac = (idx - lo.float()).unsqueeze(-1)               # (N, 1)

    rgba = (1.0 - frac) * lut[lo] + frac * lut[hi]       # (N, 4)
    return rgba.reshape(*orig_shape, 4)


# ---------------------------------------------------------------------------
# Opacity correction (Beer-Lambert, matches the CUDA implementation)
# ---------------------------------------------------------------------------

def opacity_correction(raw_alpha: torch.Tensor, step: float, reference_step: float = 1.0):
    """
    Corrects opacity for the actual step size relative to a reference step,
    equivalent to the CUDA opacityCorrection().

        corrected = 1 - (1 - raw_alpha) ^ (step / reference_step)

    Args:
        raw_alpha:      (...,)  alpha from the transfer function
        step:           actual marching step size in parametric t units
        reference_step: the step size at which the TFN was designed (default 1.0)
    """
    exponent = step / reference_step
    return 1.0 - (1.0 - raw_alpha.clamp(0.0, 1.0)) ** exponent

# ---------------------------------------------------------------------------
# Ray generation — perspective (non-parallel rays)
# ---------------------------------------------------------------------------

def generate_rays(camera: Camera, device: torch.device):
    """
    Returns:
        ray_origins    (H, W, 3)  — all identical for a pinhole camera
        ray_directions (H, W, 3)  — unit vectors, one per pixel
    """
    H, W = camera.height, camera.width

    # Build orthonormal camera basis
    forward = F.normalize(camera.look_at - camera.position, dim=0)
    if abs(torch.dot(forward, camera.up).item()) > 0.99:
        camera.up = torch.tensor([0.0, 0.0, 1.0])
    right   = F.normalize(torch.linalg.cross(forward, camera.up), dim=0)
    up_     = torch.linalg.cross(right, forward)          # reorthogonalise

    # Focal length from vertical FoV
    fov_rad    = torch.tensor(camera.fov_y * 3.14159265 / 180.0)
    focal      = 1.0 / torch.tan(fov_rad / 2.0)           # in NDC units
    aspect     = W / H

    # Pixel centres in NDC  [-1, 1]
    ys = torch.linspace( 1 - 1/H, -1 + 1/H, H, device=device)  # top→bottom
    xs = torch.linspace(-1 + 1/W,  1 - 1/W, W, device=device)

    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')       # (H, W)

    # Direction in camera space, then rotate to world space
    dir_cam = (
          grid_x.unsqueeze(-1) * aspect * right.to(device)
        + grid_y.unsqueeze(-1) * up_.to(device)
        + focal                * forward.to(device)
    )                                                              # (H, W, 3)
    ray_directions = F.normalize(dir_cam, dim=-1)

    ray_origins = camera.position.to(device).expand(H, W, 3).clone()
    
    return ray_origins, ray_directions


# ---------------------------------------------------------------------------
# Trilinear sampler for arbitrary (D, H, W, C) volumes
# ---------------------------------------------------------------------------
# NOTE: not sure why scalar values sampled from this function can not produce correct GT rendered images
# TODO: need to carefully verify the correctness of this function, and how it is different from our sampler
def sample_volume_trilinear(
    volume: torch.Tensor,   # (D, H, W, C)  — last dim is channels
    points: torch.Tensor,   # (..., 3)       — coords in [0, 1]^3  (x, y, z)
) -> torch.Tensor:          # (..., C)
    """
    Trilinear interpolation via F.grid_sample.
    Coords outside [0,1] are handled with border padding.

    volume coordinate convention: x→W, y→H, z→D
    """
    D, H, W, C = volume.shape
    orig_shape  = points.shape[:-1]

    # grid_sample expects (N, C, D, H, W) and grid in [-1, 1]
    vol = volume.permute(3, 0, 1, 2).unsqueeze(0)   # (1, C, D, H, W) — swap dim C to the second dim
    vol = vol.float()

    # Remap [0,1] → [-1, 1]
    grid = (points.reshape(1, -1, 1, 1, 3).mul_(2).sub_(1)).float()   # (1, N, 1, 1, 3)
    sampled = F.grid_sample(
        vol, grid,
        mode='bilinear',          # trilinear in 5-D
        padding_mode='zeros',     # set to align with dvnr sampler
        align_corners=False,
    )                              # (1, C, N, 1, 1)
    
    sampled = sampled.squeeze(0).squeeze(-1).squeeze(-1)   # (C, N)
    return sampled.T.reshape(*orig_shape, C)               # (..., C)


# ---------------------------------------------------------------------------
# Core ray marcher
# ---------------------------------------------------------------------------

def ray_march(
    ray_origins:    torch.Tensor,   # (H, W, 3)
    ray_directions: torch.Tensor,   # (H, W, 3)
    sampler:        Any,
    tfn_lut:        torch.Tensor,   # (lut_size, 4)
    cfg:            MarchConfig,
    tfn_file:       str,
    scene_aabb:     Optional[torch.Tensor] = None,  # (2, 3) [min, max] in world
    light_dir_normalized: list = [0.25, 0.25],
    nets: Any = None,   # if nets is not provided, fallback to use GT shadow sampler
    # volume_tensor: Any = None
):
    """
    Volume render with naive shadow blending.

    For each ray:
      1. Sample N points along the ray (perspective: each ray has a unique dir)
      2. Trilinearly interpolate density and shadow at each sample
      3. Convert density to alpha via Beer–Lambert
      4. Alpha-composite shadow coefficients along the ray

    Returns a dict with:
      'rendered_shadow'  (H, W, S)  — composited shadow value per pixel
      'opacity'          (H, W, 1)  — accumulated opacity (depth proxy)
      'depth'            (H, W, 1)  — expected depth
    """
    H, W, _ = ray_origins.shape
    device   = ray_origins.device

    # TODO: make the below iteratively for training!

    # --- Sample t values along each ray ---
    t_vals = torch.linspace(cfg.t_near, cfg.t_far, cfg.n_samples, device=device)
    # Perturb samples slightly (optional, helps reduce banding)
    if cfg.n_samples > 1:
        dt = (cfg.t_far - cfg.t_near) / cfg.n_samples
        noise = torch.rand(H, W, cfg.n_samples, device=device) * dt
        t_vals = t_vals.unsqueeze(0).unsqueeze(0) + noise   # (H, W, N)
    else:
        t_vals = t_vals.unsqueeze(0).unsqueeze(0).expand(H, W, -1)

    # TODO: sample points are generated between t_near and t_far
    # if the points fall outside t_near and t_far -> waste
    # should be able to determime more compact t_near and t_far for bounding box
    # --- World-space sample positions ---
    # origins: (H, W, 1, 3),  directions: (H, W, 1, 3),  t: (H, W, N, 1)
    pts = (ray_origins.unsqueeze(2)
           + ray_directions.unsqueeze(2) * t_vals.unsqueeze(-1))  # (H, W, N, 3)
    
    # --- Map world coords to [0,1] volume space via AABB ---
    if scene_aabb is not None:
        aabb_min = scene_aabb[0].to(device)   # (3,)
        aabb_max = scene_aabb[1].to(device)   # (3,)
    else:
        # Fallback: assume unit cube [0,1]^3
        aabb_min = torch.zeros(3, device=device)
        aabb_max = torch.ones(3,  device=device)

    # 
    pts_norm = (pts - aabb_min) / (aabb_max - aabb_min + 1e-8)   # (H, W, N, 3)
    pts_flat = pts_norm.reshape(-1, 3)                             # (H*W*N, 3)

    # -- mask: True for points inside the bounding box [0, 1]^3 --
    inside_mask = ((pts_flat >= 0.0) & (pts_flat <= 1.0)).all(dim=-1)
    
    # --- No Trilinear interpolation, instead using shadow sampler
    density_flat = torch.zeros([pts_flat.shape[0], 1], device=device)   # (H*W*N, 1)
    shadow_flat  = torch.zeros([pts_flat.shape[0], 1], device=device)   # (H*W*N, S)
    
    decode(sampler, pts_flat, density_flat)
    # density_flat = sample_volume_trilinear(volume_tensor, pts_flat)
    if nets is None:
        decode_shadow(sampler, pts_flat, shadow_flat, light_dir_normalized, tfn_file)
    else:
        shadow_flat = nets(pts_flat)

    # zero out any outside points that decode might have affected
    density_flat[~inside_mask] = 0.0
    shadow_flat[~inside_mask]  = 0.0
    
    del inside_mask
    torch.cuda.empty_cache()

    density = density_flat.reshape(H, W, cfg.n_samples, 1)          # (H, W, N, 1)
    shadow  = shadow_flat.reshape( H, W, cfg.n_samples, 1)          # (H, W, N, 1)

    # -- transfer function lookup --
    # NOTE: memory usage would significantly increase after tfn sampling
    # each of scalar values would become rgba (4 values)
    rgba    = sample_transfer_function(tfn_lut, density)          # (H, W, N, 4)
    rgb     = rgba[..., :3]                                      # (H, W, N, 3)
    alpha   = rgba[..., 3]                                       # (H, W, N)

    # -- opacity correction for actual step size --
    # step_size = (cfg.t_far - cfg.t_near) / cfg.n_samples
    # alpha   = opacity_correction(alpha, step=step_size)          # (H, W, N)

    # -- shadow blending: modulate rgb by shadow coefficient --
    S_coef  = shadow[..., 0].clamp(0.0, 1.0)                    # (H, W, N)  scalar occlusion
    ambient = 1.4

    rgb = rgb.squeeze(3)
    rgb     = torch.lerp(rgb * ambient,
                         rgb * ambient * S_coef.unsqueeze(-1),
                         0.9)                   # (H, W, N, 3)


    alpha_c = alpha
    # transmittance: T_i = prod_{j < i} (1 - alpha_c_j)
    # i.e. how much light survives before reaching sample i
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones(H, W, 1, 1, device=device),                  # T_0 = 1 (no occlusion yet)
            1.0 - alpha_c + 1e-10                                    # (H, W, N, 1)
        ], dim=2),
        dim=2
    )[:, :, :-1, :]                                                  # (H, W, N, 1) drop the last
    
    weights = transmittance * alpha_c                                # (H, W, N, 1)

    # -- final compositing --
    rendered_rgb    = (weights * rgb).sum(dim=2)                     # (H, W, 3)
    # rendered_shadow = (weights * shadow).sum(dim=2)                  # (H, W, S)
    # opacity         = weights.sum(dim=2)                             # (H, W, 1)
    # depth           = (weights * t_vals.unsqueeze(-1)).sum(dim=2)    # (H, W, 1)
    return rendered_rgb
    return {
        'rendered_rgb':    rendered_rgb,     # (H, W, 3)
        # 'rendered_shadow': rendered_shadow,  # (H, W, S)
        # 'opacity':         opacity,          # (H, W, 1)
        # 'depth':           depth,            # (H, W, 1)
        # 'weights':         weights,          # (H, W, N, 1)
    }

def test_whether_ray_start_in_volume(ray_origins, ray_directions, scene_aabb, cfg):
    
    first_hit = ray_origins + ray_directions * cfg.t_near

    scene_aabb = scene_aabb.to(ray_origins.device)

    inside_mask = ((first_hit >= scene_aabb[0]) &
               (first_hit <= scene_aabb[1])).all(dim=-1)
    
    # if any first hit points are inside the volume, return true
    return inside_mask.any()


# ---------------------------------------------------------------------------
# Full render pass
# ---------------------------------------------------------------------------

def render(
    camera:       Camera,
    sampler:      Any,
    tfn_lut:      torch.Tensor,   # (lut_size, 4)
    scene_aabb:   torch.Tensor,    # (2, 3) world bounding box
    cfg:          MarchConfig,
    device:       torch.device,
    tfn_file:     str,
    light_dir_normalized: list,
    nets: Any,
    # volume_tensor: Any,
):
    """
    End-to-end render.  Loads placeholder volumes, generates rays, ray-marches.
    """
    
    ray_origins, ray_directions = generate_rays(camera, device)

    if test_whether_ray_start_in_volume(ray_origins, ray_directions, scene_aabb, cfg):
        print("The ray sampled points start in volumes, which means near clipping plane might clip the volume.")

    result = ray_march(
        ray_origins    = ray_origins,
        ray_directions = ray_directions,
        sampler        = sampler,
        tfn_lut        = tfn_lut,
        cfg            = cfg,
        tfn_file       = tfn_file,
        scene_aabb     = scene_aabb,
        light_dir_normalized=light_dir_normalized,
        nets = nets,
        # volume_tensor=volume_tensor,
    )
    return result

def ray_march_with_precalculated_pts(
        ray_sampled_pts:    torch.Tensor,   # (n_rays, N, 4)
        inside_mask:        torch.Tensor,   # (n_rays, N)
        sampler:            Any,
        tfn_lut:            torch.Tensor,   # (lut_size, 4)
        cfg:                MarchConfig,
        tfn_file:           str,
        light_dir_normalized: list = [0.25, 0.25],
        net:                Any = None,
    ):
        device = ray_sampled_pts.device
        
        pts_flat = ray_sampled_pts.reshape(-1, 4)   # (n_rays*N, 4)
        inside_mask = inside_mask.flatten()     # (n_rays*N)
        
        # density_flat = torch.zeros([pts_flat.shape[0], 1], device=device)   # (n_rays*N, 1)
        # decode(self.sampler, pts_flat, density_flat)
        
        density_flat = pts_flat[:, 3:]      # (n_rays*N, 1)
        pts_coords = pts_flat[:, :3].clone()
        shadow_flat  = torch.zeros([pts_flat.shape[0], 1], device=device)   # (n_rays*N, 1)
        if net is None:
            decode_shadow(sampler, pts_coords, shadow_flat, light_dir_normalized, tfn_file)
        else:
            with torch.no_grad():
                C = 65536
                for i in range(0, pts_coords.shape[0], C):
                    shadow_flat[i:i+C] = net(pts_coords[i:i+C])
        
        # zero out any outside points that decode might have affected
        # density_flat[~inside_mask] = 0.0
        shadow_flat[~inside_mask]  = 0.0
        
        # del inside_mask
        # torch.cuda.empty_cache()

        density = density_flat.reshape(-1, cfg.n_samples, 1)          # (n_rays, N, 1)
        shadow  = shadow_flat.reshape(-1, cfg.n_samples, 1)          # (n_rays, N, 1)

        # -- transfer function lookup --
        rgba    = sample_transfer_function(tfn_lut, density)    # (n_rays, N, 1, 4)
        rgba    = rgba.squeeze(2)                                    # (n_rays, N, 4)
        rgb     = rgba[..., :3]                                      # (n_rays, N, 3)
        alpha   = rgba[..., 3:]                                       # (n_rays, N, 1)

        # -- opacity correction for actual step size --
        # step_size = (cfg.t_far - cfg.t_near) / cfg.n_samples
        # alpha   = opacity_correction(alpha, step=step_size)        # (H, W, N)

        # -- shadow blending: modulate rgb by shadow coefficient --
        shadow = shadow.clamp(0.0, 1.0)             # (n_rays, N, 1)
        ambient = 1.4
        
        # should copy rgb n_batch times
        rgb = torch.lerp(rgb * ambient,
                        rgb * ambient * shadow,
                        0.9)                   # (n_rays, N, 3)
        
        alpha_c = alpha     # (n_rays, N, 1)
        
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones(alpha_c.shape[0], 1, 1, device=device), # T_0 = 1 (no occlusion yet)
                1.0 - alpha_c + 1e-10                                    # (n_rays, N, 1)
            ], dim=1),
            dim=1
        )[:, :-1, :]                                                  # (n_rays, N, 1) drop the last
        
        weights = transmittance * alpha_c                             # (n_rays, N, 1)

        # -- final compositing --
        rendered_rgb = (weights * rgb).sum(dim=1)          # (n_rays, 3)
        return rendered_rgb

# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cam = Camera(
        # TODO: see how can I convert camera parameters from world space to object space
        # for example, the object (mechhand) in world space is (0,0,0)~(640,220,229), and its object space is (0,0,0)~(1,1,1)
        # for mechhand
        # position = torch.tensor([0.0, 1.5,  1.5]),
        # look_at  = torch.tensor([0.5, 0.5,  0.5]),
        # up       = torch.tensor([0.0, 1.0,  0.0]),
        # for spider
        position = torch.tensor([-0.085, -0.31, -0.012]),
        look_at  = torch.tensor([0.45, 0.4,  0.5]),
        up       = torch.tensor([0.1, 0.342,  -0.93]),
        # position = torch.tensor([-1.0394, 2.5554, 2.5044]),
        # look_at  = torch.tensor([0.5-0.004525, 0.5-0.3158,  1.1093]),
        # up       = torch.tensor([0.3800, 0.8572,  -0.341]),
        fov_y    = 60.0,
        width    = 384,
        height   = 384,
    )

    aabb = torch.tensor([[0., 0., 0.], [1., 1., 1.]])

    cfg = MarchConfig(
        # for mechhand
        # t_near    = 0.1,
        # t_far     = 2.8,
        # n_samples = 1024,
        # for spider
        # NOTE: might be okay to clip volume a little bit as we just want the majority of the rendered volume for calculating rendering loss
        t_near    = 0.6,
        t_far     = 1.3,
        n_samples = 1024,
        patch_width=16,
        patch_height=16,
    )

    # raw_data_file_path = "/home/kctung/Projects/BlockFusion/datasets/MechHand_f_640x220x229.raw"
    # tfn_file_path = "/home/kctung/Projects/BlockFusion/datasets/scene_mechhand.json"
    raw_data_file_path = "/home/kctung/Projects/BlockFusion/datasets/zea_c_spider2_957x1195x1003_float32.raw"
    tfn_file_path = "/home/kctung/Projects/BlockFusion/datasets/scene_spider_transparent_pink.json"

    # sample light direction normalized
    # for mechhand 
    # light_dir_normalized = [0.25, 0.25]
    # for spider
    # light_dir_normalized = [0.76, 0.4]
    light_dir_normalized = [0.555, 0.827]

    with open(tfn_file_path, 'r') as f:
        tfn_json = json5.load(f)
    
    resolution = tfn_json["dataSource"][0]["dimensions"]
    resolution = [resolution["x"], resolution["y"], resolution["z"]]
    colorControls = tfn_json["view"]["volume"]["transferFunction"]["colorControls"]
    opacityControl = tfn_json["view"]["volume"]["transferFunction"]["opacityControl"]

    sampler = create_sampler("structuredRegular", "cuda", dims=resolution, dtype="float32", n_channels=1, filename=raw_data_file_path)

    with open("shadows_subset_training_SIREN.json", 'r') as f:
        config = json5.load(f)
    config = edict(config)
        
    loaded_model = torch.load("/home/kctung/Projects/HyperDiffusion/logs/spider_2000_embd720_head16_layer12/2026-02-25_06-02-52/sample_siren_7000_test.pt", map_location="cpu")
    
    n_instances = len(loaded_model['light_dir_cartesian'])
    
    nets = [NeurCompNet(n_input_dims=3, 
                    n_output_dims=config.n_labels, bias=False, 
                    n_hidden_layers=config.n_layers, 
                    n_neurons=config.n_hid, is_residual=True).cuda() for _ in range(n_instances)]
    nets = torch.nn.ModuleList(nets)
    nets.load_state_dict(loaded_model['net_state_dict'])
    nets.eval()
    nets = nets[0]

    tfn_lut = build_transfer_function(colorControls, opacityControl, lut_size=1024)
    # result = patchify_render_training(cam, sampler, tfn_lut, scene_aabb=aabb, cfg=cfg, device=device, GT_image=torch.zeros([cam.height, cam.width, 3], device=device, dtype=torch.float32), nets=nets)
    # render GT image
    result = render(cam, sampler, tfn_lut, scene_aabb=aabb, cfg=cfg, device=device, tfn_file=tfn_file_path, light_dir_normalized=light_dir_normalized, nets=None)
    # import pdb; pdb.set_trace()
    # print("rendered_shadow:", result['rendered_shadow'].shape)   # (64, 64, 1)
    # print("opacity:        ", result['opacity'].shape)           # (64, 64, 1)
    # print("depth:          ", result['depth'].shape)             # (64, 64, 1)
    print("Smoke test passed.")
    print(f"max memory allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"max memory reserved: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
    
    save_dir = "/home/kctung/Projects/BlockFusion/logs/rendering_loss_dev_exp"
    
    data = result
    plt.figure(figsize=(7, 7))
    # Convert to numpy if tensor
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    # Plot with colorbar
    if data.shape[-1] == 3:
        im = plt.imshow(data)
    else:
        im = plt.imshow(data, cmap='viridis')
    # cbar = plt.colorbar()
    plt.title("Full render with shadow")
    plt.savefig(os.path.join(save_dir,"Full_render_with_shadow.png"))
    plt.close()
    
    # to generate GT data
    result_to_save = dict()
    # HACK: only one sample now
    result_to_save["GT_image"] = result
    result_to_save["light_dir_spherical_normalized"] = light_dir_normalized
    # save image tensor as pt file
    torch.save(result_to_save, os.path.join(save_dir, "GT_image.pt"))
