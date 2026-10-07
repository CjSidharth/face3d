"""Unit tests for pipeline/printability.py using synthetic meshes.

These deliberately avoid running the full TripoSR model (slow, needs downloaded
weights) — instead they build a simple procedural "bust-like" shape (a sphere head
atop a wider cylinder for shoulders), rotate it into an arbitrary orientation, and
inject the kinds of defects a raw reconstruction can have (an open hole, a stray
disconnected fragment), to validate the repair/orient/base/pedestal contract in
isolation.
"""
import numpy as np
import pytest
import trimesh

from app.config import settings
from pipeline.printability import (
    PrintabilityError,
    add_pedestal,
    close_base,
    make_printable,
    orient_upright,
    repair_mesh,
    scale_to_height,
)


def _make_messy_bust() -> trimesh.Trimesh:
    """A head (small sphere) fused onto shoulders (wide cylinder), rotated into an
    arbitrary orientation, with an injected hole and a disconnected fragment —
    i.e. a stand-in for a raw, imperfect reconstruction."""
    head = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    head.apply_translation([0, 0, 12])
    shoulders = trimesh.creation.cylinder(radius=10, height=10, sections=32)
    shoulders.apply_translation([0, 0, 3])
    bust = trimesh.boolean.union([head, shoulders], engine="manifold")
    assert bust.is_watertight  # sanity check on the test fixture itself

    # Arbitrary rotation so "up" is not already +Z — proves orientation is derived
    # from geometry, not assumed.
    rotation = trimesh.transformations.euler_matrix(0.7, 1.1, -0.4)
    bust.apply_transform(rotation)

    # Punch a hole: drop a handful of faces from a non-boundary region.
    keep = np.ones(len(bust.faces), dtype=bool)
    keep[10:14] = False
    bust.update_faces(keep)

    # Add a small disconnected fragment far away, like a marching-cubes artifact.
    fragment = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    fragment.apply_translation([200, 200, 200])
    messy = trimesh.util.concatenate([bust, fragment])
    assert not messy.is_watertight  # sanity check: the defects took effect
    return messy


def test_repair_mesh_produces_single_component_manifold():
    mesh = repair_mesh(_make_messy_bust())
    # keep-largest-component should have dropped the far-away fragment.
    assert mesh.vertices.shape[0] > 0
    components = mesh.split(only_watertight=False)
    assert len(components) == 1


def test_orient_upright_puts_head_on_top():
    mesh = repair_mesh(_make_messy_bust())
    mesh = orient_upright(mesh)
    z = mesh.vertices[:, 2]
    z_min, z_max = z.min(), z.max()
    top = mesh.vertices[z >= np.percentile(z, 90)]
    bottom = mesh.vertices[z <= np.percentile(z, 10)]
    top_spread = np.linalg.norm(top[:, :2] - top[:, :2].mean(axis=0), axis=1).std()
    bottom_spread = np.linalg.norm(bottom[:, :2] - bottom[:, :2].mean(axis=0), axis=1).std()
    # Head (top) should be narrower than shoulders (bottom).
    assert top_spread < bottom_spread
    assert z_max > z_min


def test_close_base_produces_flat_bottom_and_stays_watertight():
    mesh = orient_upright(repair_mesh(_make_messy_bust()))
    capped = close_base(mesh, settings.base_trim_fraction)
    z_min = capped.bounds[0][2]
    bottom_verts = capped.vertices[capped.vertices[:, 2] <= z_min + 1e-6]
    assert bottom_verts.shape[0] >= 3
    assert capped.is_watertight


def test_scale_to_height_hits_target():
    mesh = close_base(orient_upright(repair_mesh(_make_messy_bust())), settings.base_trim_fraction)
    scaled = scale_to_height(mesh, 42.0)
    height = scaled.bounds[1][2] - scaled.bounds[0][2]
    assert height == pytest.approx(42.0, rel=1e-3)


def test_add_pedestal_is_wider_and_watertight():
    mesh = scale_to_height(
        close_base(orient_upright(repair_mesh(_make_messy_bust())), settings.base_trim_fraction),
        100.0,
    )
    fused = add_pedestal(mesh, pedestal_height_mm=8.0, radius_ratio=1.15, overlap_mm=2.0)
    assert fused.is_watertight
    assert fused.is_winding_consistent
    total_height = fused.bounds[1][2] - fused.bounds[0][2]
    # pedestal adds height on top of the bust's own 100mm
    assert total_height > 100.0


def test_make_printable_end_to_end():
    mesh, report = make_printable(_make_messy_bust(), height_mm=100.0)
    assert report.is_watertight
    assert report.is_winding_consistent
    assert report.volume_mm3 > 0
    assert report.bust_height_mm == pytest.approx(100.0, rel=1e-2)
    assert report.total_height_mm > report.bust_height_mm
    # Final mesh should sit on the print bed.
    assert mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6)


def test_make_printable_rejects_nonpositive_height():
    with pytest.raises(PrintabilityError):
        make_printable(_make_messy_bust(), height_mm=0)


def test_repair_mesh_rejects_empty_input():
    empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
    with pytest.raises(PrintabilityError):
        repair_mesh(empty)
