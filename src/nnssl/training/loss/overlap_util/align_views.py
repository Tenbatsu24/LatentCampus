from typing import Tuple, Union

import torch
import torch.nn.functional as F


def align_views(
    latents: torch.Tensor,  # [B, C, D, H, W]
    rel_bboxes: torch.Tensor,  # [B, 6] – (x1, y1, z1, x2, y2, z2) in [0, 1]
    out_size: Union[int, Tuple[int, int, int]] | None = None,
    sampling_ratio: float = None,
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    ROI-aligns each latent volume to its corresponding relative 3‑D bounding box.

    Args:
        latents:     5-D tensor with shape **[B, C, D, H, W]**.
        rel_bboxes:  Normalised boxes per sample **[B, 6]** –
                     (x1, y1, z1, x2, y2, z2), all ∈ [0, 1].
        out_size:    Target crop size. If:
                        • None  → keep original (D, H, W)
                        • int   → cubic volume `(out_size, out_size, out_size)`
                        • tuple → explicit `(D_out, H_out, W_out)`
        sampling_ratio: If not None, use 2.
        padding_mode: How to handle sampling outside the input (“border”, “zeros”, “reflection”).
        align_corners: Passed to `torch.nn.functional.grid_sample`.

    Returns:
        aligned_latents: Cropped & (tri)linearly‑resampled volumes `[B, C, D_out, H_out, W_out]`.
    """
    # ── sanity checks ────────────────────────────────────────────────────────────
    if latents.ndim != 5:
        raise ValueError(f"`latents` must be 5‑D (B×C×D×H×W); got {latents.shape}.")
    if rel_bboxes.shape[-1] != 6:
        raise ValueError("`rel_bboxes` must have 6 values per box (x1,y1,z1,x2,y2,z2).")
    if rel_bboxes.size(0) != latents.size(0):
        raise ValueError("Batch size mismatch between latents and rel_bboxes.")
    if (rel_bboxes < 0).any() or (rel_bboxes > 1).any():
        raise ValueError("`rel_bboxes` values must lie in [0, 1].")

    if sampling_ratio is None:
        sampling_ratio = 2

    B, C, D, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # ── determine output resolution ──────────────────────────────────────────────
    if out_size is None:
        D_out, H_out, W_out = D, H, W
    elif isinstance(out_size, int):
        D_out = H_out = W_out = out_size
    else:
        D_out, H_out, W_out = out_size

    # ── build a base grid in [-1, 1] ────────────────────────────────────────────
    z_lin = torch.linspace(-1, 1, sampling_ratio * D_out, device=device, dtype=dtype)
    y_lin = torch.linspace(-1, 1, sampling_ratio * H_out, device=device, dtype=dtype)
    x_lin = torch.linspace(-1, 1, sampling_ratio * W_out, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z_lin, y_lin, x_lin, indexing='ij')  # (D_out, H_out, W_out)
    base_grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)  # (1, D_out, H_out, W_out, 3)

    # ── prepare per‑sample scale & shift ───────────────────────────────────────
    x1, y1, z1, x2, y2, z2 = rel_bboxes.unbind(dim=-1)  # each (B,)
    # centre in [0,1], size in [0,1]
    cx, cy, cz = (x1 + x2) * 0.5, (y1 + y2) * 0.5, (z1 + z2) * 0.5
    sx, sy, sz = (x2 - x1), (y2 - y1), (z2 - z1)

    # convert to shift / scale for [-1,1] space
    shift = torch.stack((2 * cx - 1, 2 * cy - 1, 2 * cz - 1), dim=-1)  # (B, 3)
    scale = torch.stack((sx, sy, sz), dim=-1)  # (B, 3)

    shift = shift.view(B, 1, 1, 1, 3)
    scale = scale.view(B, 1, 1, 1, 3)

    # ── produce the sampling grid ───────────────────────────────────────────────
    sampling_grid = base_grid * scale + shift  # (B, D_out, H_out, W_out, 3)

    # ── trilinear ROI‑align via grid_sample ────────────────────────────────────
    aligned_latents = F.grid_sample(
        latents, sampling_grid,
        mode="bilinear",  # 5‑D “bilinear” = trilinear
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # adaptive pool to the ouput size
    aligned_latents = F.adaptive_avg_pool3d(aligned_latents, (D_out, H_out, W_out))

    return aligned_latents


if __name__ == '__main__':
    # Create random 3D latent volumes: [batch=2, channels=1, depth=32, height=32, width=32]
    _latents = torch.rand(2, 1, 32, 32, 32)

    # Define random relative bounding boxes in [0, 1]
    # Format: (x1, y1, z1, x2, y2, z2)
    _rel_bboxes = torch.tensor([
        [0.2, 0.2, 0.2, 0.8, 0.8, 0.8],  # centered box
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],  # corner crop
    ], dtype=torch.float32)

    # Run the alignment
    _crops = align_views(_latents, _rel_bboxes, out_size=16)

    # Print results
    print("Input volume shape:", _latents.shape)
    print("Aligned crops shape:", _crops.shape)  # Expect: [2, 1, 16, 16, 16]
