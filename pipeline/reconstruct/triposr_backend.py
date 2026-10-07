"""TripoSR implementation of the ReconstructionBackend protocol.

Runs on MPS (falling back to CPU per-op via PYTORCH_ENABLE_MPS_FALLBACK, set in
app.config) — never CUDA. Marching cubes uses scikit-image, not the upstream
`torchmcubes` dependency; see pipeline/reconstruct/vendor/tsr/README_VENDOR.md.
"""
import sys
from pathlib import Path

import PIL.Image
import trimesh

from app.config import settings  # also sets HF_HOME/U2NET_HOME as an import side effect

# `tsr`'s own modules do absolute imports (importlib.import_module("tsr....")), so it
# needs to be importable as a top-level package — add the vendor directory that
# contains it to sys.path rather than importing it as a submodule of this package.
_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tsr.system import TSR  # noqa: E402
from tsr.utils import remove_background as tsr_remove_background  # noqa: E402
from tsr.utils import resize_foreground  # noqa: E402


class ReconstructionError(RuntimeError):
    """Raised when TripoSR fails to produce a usable mesh from the input image."""


class TripoSRBackend:
    """Loads TripoSR once and reuses it across requests (weights are ~hundreds of MB;
    do not construct a new instance per request)."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        chunk_size: int | None = None,
        mc_resolution: int | None = None,
        foreground_ratio: float | None = None,
    ) -> None:
        self.model_id = model_id or settings.triposr_model_id
        self.device = device or settings.device
        self.chunk_size = chunk_size or settings.triposr_chunk_size
        self.mc_resolution = mc_resolution or settings.marching_cubes_resolution
        self.foreground_ratio = foreground_ratio or settings.foreground_ratio
        self._model: TSR | None = None

    @property
    def model(self) -> TSR:
        if self._model is None:
            model = TSR.from_pretrained(
                self.model_id,
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            model.renderer.set_chunk_size(self.chunk_size)
            model.to(self.device)
            self._model = model
        return self._model

    def _format_for_triposr(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """Replicate TripoSR's expected input formatting: RGBA cutout -> centered,
        padded square at `foreground_ratio` -> composited onto flat 50% gray (the
        background TripoSR was trained on)."""
        # No-op if `image` already has real transparency (the normal case, since
        # pipeline.preprocess.preprocess already ran rembg); only removes background
        # itself if handed a flat RGB image directly.
        image = tsr_remove_background(image)
        image = resize_foreground(image, self.foreground_ratio)

        arr = np.array(image).astype(np.float32) / 255.0
        if arr.shape[-1] != 4:
            raise ReconstructionError(
                "Expected an RGBA image with alpha transparency after background "
                "removal; got shape "
                f"{arr.shape}. Preprocessing likely failed upstream."
            )
        rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        return PIL.Image.fromarray((rgb * 255.0).astype(np.uint8))

    def reconstruct(self, image: PIL.Image.Image) -> trimesh.Trimesh:
        formatted = self._format_for_triposr(image)

        with torch.no_grad():
            scene_codes = self.model([formatted], device=self.device)
            meshes = self.model.extract_mesh(
                scene_codes, has_vertex_color=False, resolution=self.mc_resolution
            )

        if not meshes or meshes[0].vertices.shape[0] == 0:
            raise ReconstructionError(
                "TripoSR produced an empty mesh. This can happen with low-contrast "
                "or ambiguous silhouettes after background removal — try a photo "
                "with a clearer, more even background."
            )
        return meshes[0]
