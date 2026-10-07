"""Write a mesh to STL and verify it's actually watertight on disk.

Verifying the in-memory `trimesh.Trimesh` isn't enough — STL export/import can
itself introduce precision issues (STL stores raw triangle soup, no shared-vertex
topology), so this re-loads the written file and re-checks it independently. That
re-check is the actual guarantee this project promises ("export a valid STL and
verify it's watertight"), not just an assumption carried over from the mesh in memory.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import trimesh


class ExportError(RuntimeError):
    """Raised when the exported STL fails watertight/manifold verification."""


@dataclass
class ExportReport:
    path: str
    is_watertight: bool
    is_winding_consistent: bool
    euler_number: int
    volume_mm3: float
    face_count: int
    vertex_count: int


def export_stl(mesh: trimesh.Trimesh, path: Path, *, require_watertight: bool = True) -> ExportReport:
    """Write `mesh` to `path` as binary STL, then re-load and verify it.

    Raises `ExportError` if `require_watertight` and the re-loaded STL isn't a
    watertight, consistently-wound solid with positive volume — this is meant to be
    a hard gate before a file is handed back to the caller as "printable".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path, file_type="stl")

    # Re-load independently rather than trusting the in-memory mesh we just wrote.
    reloaded = trimesh.load(path, file_type="stl", process=True)
    if not isinstance(reloaded, trimesh.Trimesh):
        raise ExportError(f"Re-loading {path} did not produce a single mesh (got {type(reloaded)}).")

    watertight = bool(reloaded.is_watertight)
    winding_ok = bool(reloaded.is_winding_consistent)
    volume = float(reloaded.volume) if watertight else float("nan")

    report = ExportReport(
        path=str(path),
        is_watertight=watertight,
        is_winding_consistent=winding_ok,
        euler_number=int(reloaded.euler_number),
        volume_mm3=volume,
        face_count=int(reloaded.faces.shape[0]),
        vertex_count=int(reloaded.vertices.shape[0]),
    )

    if require_watertight and not (watertight and winding_ok and volume > 0):
        raise ExportError(
            f"Exported STL at {path} failed verification: "
            f"watertight={watertight}, winding_consistent={winding_ok}, volume={volume}. "
            "Report: " + str(asdict(report))
        )

    return report
