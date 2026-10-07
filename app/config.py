"""Central settings for the face3d pipeline and API.

Apple Silicon note: this MUST be imported (or PYTORCH_ENABLE_MPS_FALLBACK set) before
`torch` touches any MPS op, so the fallback-to-CPU env var is set as a side effect of
importing this module, at process start.
"""
import os
from pathlib import Path

# Must be set before any MPS op runs; a handful of ops used by TripoSR's transformer
# backbone / renderer are not yet implemented on MPS and need the CPU fallback.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (import after env var is set)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep every downloaded model weight under models/ instead of scattering into
# per-library default caches (~/.cache/huggingface, ~/.u2net, ...). TripoSR's own
# hf_hub_download calls respect HF_HOME; rembg respects U2NET_HOME. Both must be set
# before those libraries are imported anywhere.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf_cache"))
os.environ.setdefault("U2NET_HOME", str(MODELS_DIR / "u2net"))
# huggingface_hub's newer "Xet" CDN backend has been observed to stall/500-loop on
# some networks; the plain HTTP downloader is slower but far more reliable for a
# one-time weight download. Force it off rather than hang indefinitely on retries.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def resolve_device() -> str:
    """Pick the best available device: MPS > CPU. No CUDA path in this project."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Settings:
    # --- Reconstruction ---
    device: str = resolve_device()
    triposr_model_id: str = "stabilityai/TripoSR"
    triposr_chunk_size: int = 8192
    # 256 leaves visible voxel/staircase banding on the surface (marching cubes grid
    # size). 384 gives noticeably finer geometry for a modest runtime cost on MPS.
    marching_cubes_resolution: int = 384
    foreground_ratio: float = 0.85  # how much of the frame the subject fills after crop

    # --- Printability ---
    # `default_height_mm` is the bust's own height (head+shoulders, after the base
    # trim below); the pedestal adds `pedestal_height_mm` on top of that for the
    # total printed height. See pipeline/printability.py for the full rationale.
    default_height_mm: float = 100.0
    base_trim_fraction: float = 0.03  # trim this fraction off the bottom before capping flat
    pedestal_height_mm: float = 8.0
    pedestal_radius_ratio: float = 1.15  # pedestal radius as a multiple of the bust's base radius (>1 = stable, visible lip)
    pedestal_overlap_mm: float = 2.0  # pedestal reaches this far up into the bust so the boolean union has true volumetric overlap, not just a coincident face
    hole_fill_max_hole_size: int = 10_000  # trimesh/pymeshlab hole-filling cap (triangles)
    # Taubin smoothing (alternating shrink/inflate passes) removes marching-cubes
    # voxel banding without the shrinkage/blurring plain Laplacian smoothing causes.
    # See pipeline/printability.py:_pymeshlab_repair.
    smoothing_steps: int = 8
    smoothing_lambda: float = 0.5
    smoothing_mu: float = -0.53

    # --- Storage ---
    models_dir: Path = MODELS_DIR
    outputs_dir: Path = OUTPUTS_DIR

    # --- API ---
    max_upload_bytes: int = 20 * 1024 * 1024  # 20 MB
    request_timeout_s: int = 300  # reconstruction on MPS/CPU is ~30-90s; generous margin


settings = Settings()
