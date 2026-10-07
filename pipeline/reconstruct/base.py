"""The reconstruction seam: every single-image-to-3D model implements this.

Downstream (`pipeline/printability.py`, `pipeline/export.py`) only ever sees a plain
`trimesh.Trimesh` — they have no knowledge of which backend produced it. This is what
lets the MVP ship with TripoSR (MPS-native, MIT-licensed) while leaving room to swap in
a FLAME-based model or a remote-GPU backend later without touching anything else in
the pipeline. See PLAN.md's model evaluation table for why TripoSR was chosen first.
"""
from typing import Protocol

import PIL.Image
import trimesh


class ReconstructionBackend(Protocol):
    def reconstruct(self, image: PIL.Image.Image) -> trimesh.Trimesh:
        """Turn a preprocessed (cropped, background-removed) RGBA image into a mesh.

        The input is expected to already have passed through
        `pipeline.preprocess.preprocess` — this method should not assume it needs to
        do its own face detection or cropping, though it may do backend-specific
        final formatting (e.g. compositing onto a particular background color).

        Returns a `trimesh.Trimesh` with no texture/vertex-color requirement — the
        printability/export stages only care about geometry.
        """
        ...
