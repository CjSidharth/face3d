"""Unit tests for pipeline/export.py."""
import numpy as np
import pytest
import trimesh

from pipeline.export import ExportError, export_stl


def test_export_watertight_mesh_succeeds(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    assert mesh.is_watertight

    report = export_stl(mesh, tmp_path / "sphere.stl")

    assert report.is_watertight
    assert report.is_winding_consistent
    assert report.volume_mm3 > 0
    assert (tmp_path / "sphere.stl").exists()


def test_export_report_roundtrips_from_disk(tmp_path):
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    report = export_stl(mesh, tmp_path / "box.stl")

    reloaded = trimesh.load(tmp_path / "box.stl")
    assert reloaded.is_watertight
    assert report.volume_mm3 == pytest.approx(1000.0, rel=1e-3)
    assert report.face_count == reloaded.faces.shape[0]


def test_export_non_watertight_mesh_raises_by_default(tmp_path):
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    # Drop a face to make it non-watertight.
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[0] = False
    mesh.update_faces(keep)
    assert not mesh.is_watertight

    with pytest.raises(ExportError):
        export_stl(mesh, tmp_path / "broken.stl")


def test_export_non_watertight_mesh_allowed_when_not_required(tmp_path):
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[0] = False
    mesh.update_faces(keep)

    report = export_stl(mesh, tmp_path / "broken.stl", require_watertight=False)
    assert report.is_watertight is False
    assert (tmp_path / "broken.stl").exists()
