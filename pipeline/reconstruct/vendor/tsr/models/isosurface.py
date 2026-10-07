from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# --- PATCHED for Apple Silicon (face3d project) ---------------------------
# Upstream TripoSR uses `torchmcubes`, which fails to build (or silently
# falls back to a very slow CPU-only build) on Apple Silicon: it has no
# prebuilt wheel and its CUDA extension does not target Metal/MPS. We swap
# in scikit-image's pure-Python/C marching cubes instead, which needs no
# compilation and runs identically on any platform.
#
# Coordinate note: torchmcubes' output axis order is reversed relative to
# the input volume's own axes, which is why upstream reorders vertex
# columns with `v_pos[..., [2, 1, 0]]`. skimage.measure.marching_cubes
# returns vertices already in the same axis order as the input array
# (verts[:, i] <-> volume axis i), so that reorder is dropped here. Any
# residual orientation mismatch is harmless for this project: the
# printability stage re-orients the mesh from its geometry (PCA + face
# detection), not from an assumed marching-cubes convention.
from skimage import measure


def _skimage_marching_cubes(
    level: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    volume = level.detach().cpu().numpy().astype(np.float64)
    lo, hi = volume.min(), volume.max()
    if not (lo < threshold < hi):
        raise RuntimeError(
            "Marching cubes found no isosurface at the given threshold "
            f"(volume range [{lo:.4f}, {hi:.4f}], threshold {threshold}). "
            "This usually means the reconstructed density field is empty or "
            "the input image/background removal produced no usable subject."
        )
    verts, faces, _normals, _values = measure.marching_cubes(volume, level=threshold)
    return (
        torch.from_numpy(verts.astype(np.float32)),
        torch.from_numpy(faces.astype(np.int64)),
    )


# ---------------------------------------------------------------------------


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Callable = _skimage_marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
