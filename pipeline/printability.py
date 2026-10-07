"""Turn a raw reconstructed mesh into a watertight, manifold, printable bust.

This is the core value-add of the project (see PLAN.md). Input is whatever a
`ReconstructionBackend` produced — possibly non-manifold, with duplicate/degenerate
faces, stray floating fragments, and no notion of "up" or real-world scale. Output is
a single watertight solid: oriented upright, flat-bottomed, sitting on a pedestal,
scaled to a real-world height in millimeters.

Every function here takes/returns a plain `trimesh.Trimesh` — no backend-specific
knowledge leaks in from `pipeline/reconstruct/`.
"""
from dataclasses import dataclass

import numpy as np
import pymeshlab
import trimesh

from app.config import settings


class PrintabilityError(RuntimeError):
    """Raised when a mesh cannot be made printable (e.g. degenerates to nothing)."""


@dataclass
class PrintabilityReport:
    bust_height_mm: float
    total_height_mm: float
    is_watertight: bool
    is_winding_consistent: bool
    volume_mm3: float
    face_count: int
    vertex_count: int


# --------------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------------


def _pymeshlab_repair(
    mesh: trimesh.Trimesh,
    max_hole_size: int,
    *,
    smoothing_steps: int = 0,
    smoothing_lambda: float = 0.5,
    smoothing_mu: float = -0.53,
) -> trimesh.Trimesh:
    """Run pymeshlab's cleanup filters: dedupe, drop degenerate geometry, fix
    non-manifold edges/vertices, close remaining holes up to `max_hole_size` (in
    triangle count, per pymeshlab's own convention), then optionally smooth.

    Smoothing runs last, after holes are closed, so the new cap faces get blended
    in with the rest of the surface rather than leaving a visible seam. It uses
    Taubin smoothing (alternating shrink/inflate passes with `lambda_`/`mu`) rather
    than plain Laplacian smoothing specifically because it doesn't shrink or blur
    the mesh the way repeated Laplacian passes do — important here since the goal
    is removing marching-cubes voxel banding, not softening facial features.
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_repair_non_manifold_edges(method="Remove Faces")
    ms.meshing_close_holes(maxholesize=max_hole_size)
    if smoothing_steps > 0:
        ms.apply_coord_taubin_smoothing(
            lambda_=smoothing_lambda, mu=smoothing_mu, stepsmoothnum=smoothing_steps
        )
    out = ms.current_mesh()
    repaired = trimesh.Trimesh(
        vertices=out.vertex_matrix(), faces=out.face_matrix(), process=True
    )
    return repaired


def _keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Marching cubes / repair can leave tiny disconnected artifact fragments;
    keep only the main body."""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh
    return max(components, key=lambda m: m.volume if m.is_watertight else m.area)


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """First repair pass: get to a single, clean, (mostly) manifold component.
    Does not yet guarantee watertightness — `close_base` and the final repair pass
    in `make_printable` handle that after orientation/trimming."""
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise PrintabilityError("Input mesh is empty; nothing to repair.")

    mesh = mesh.copy()
    mesh.merge_vertices()
    # trimesh 5.x dropped the remove_duplicate_faces()/remove_degenerate_faces()
    # instance methods; nondegenerate_faces() + update_faces() is the current API.
    # Duplicate-face removal is left to pymeshlab's pass just below.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.fix_normals()
    mesh = _keep_largest_component(mesh)
    mesh = _pymeshlab_repair(
        mesh,
        settings.hole_fill_max_hole_size,
        smoothing_steps=settings.smoothing_steps,
        smoothing_lambda=settings.smoothing_lambda,
        smoothing_mu=settings.smoothing_mu,
    )
    mesh = _keep_largest_component(mesh)
    return mesh


# --------------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------------


def _pca_axes(mesh: trimesh.Trimesh) -> np.ndarray:
    """Principal axes of the vertex distribution, columns sorted by descending
    variance: column 0 is the axis of greatest spread."""
    verts = mesh.vertices - mesh.vertices.mean(axis=0)
    cov = np.cov(verts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]


def _resolve_up_sign(mesh: trimesh.Trimesh, up_axis: np.ndarray, other1: np.ndarray, other2: np.ndarray) -> float:
    """A bust is narrower at the top (head) than the bottom (shoulders). Compare the
    cross-sectional spread at the two extremes along the candidate up-axis and pick
    the sign that puts the narrower end on top."""
    verts = mesh.vertices - mesh.vertices.mean(axis=0)
    t = verts @ up_axis
    lo, hi = np.percentile(t, [10, 90])

    def spread(mask: np.ndarray) -> float:
        pts = verts[mask]
        proj = np.stack([pts @ other1, pts @ other2], axis=1)
        return float(proj.std())

    top_spread = spread(t >= hi)
    bottom_spread = spread(t <= lo)
    return 1.0 if top_spread <= bottom_spread else -1.0


def orient_upright(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate the mesh so the head-to-shoulders axis is world +Z, without assuming
    any particular reconstruction backend's camera/world convention.

    Rationale: TripoSR (like most feed-forward single-image-to-3D models) has its
    own canonical output orientation from training, which isn't documented as a
    stable contract and shouldn't be hardcoded here. Instead we derive "up" from the
    geometry itself: for a head+shoulders bust, the head-to-shoulders axis is the
    direction of greatest spread (PCA's first component), and the head end is
    narrower than the shoulders end. This also makes the pipeline robust to
    swapping in a different reconstruction backend later with a different
    convention, without touching this function.
    """
    mesh = mesh.copy()
    axes = _pca_axes(mesh)
    up = axes[:, 0]
    candidate_x = axes[:, 1]
    # Re-orthonormalize against `up` and build a right-handed basis via cross product
    # rather than trusting PCA's third eigenvector's sign, so the rotation is a pure
    # rotation (determinant +1) and never mirrors the mesh inside-out.
    candidate_x = candidate_x - np.dot(candidate_x, up) * up
    candidate_x /= np.linalg.norm(candidate_x)
    y_axis = np.cross(up, candidate_x)

    sign = _resolve_up_sign(mesh, up, candidate_x, y_axis)
    up = up * sign
    # Recompute y to keep the basis right-handed after flipping `up`.
    y_axis = np.cross(up, candidate_x)

    rotation = np.stack([candidate_x, y_axis, up], axis=0)  # rows: new x, y, z(=up)
    centroid = mesh.vertices.mean(axis=0)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    mesh.apply_transform(transform)
    return mesh


# --------------------------------------------------------------------------------
# Base / pedestal
# --------------------------------------------------------------------------------


def close_base(mesh: trimesh.Trimesh, trim_fraction: float) -> trimesh.Trimesh:
    """Slice off the bottom `trim_fraction` of the mesh's height and cap the cut
    flat. Applied unconditionally (not just when there's already a hole) so the
    base is always a clean flat plane regardless of how the reconstruction tapers
    off underneath — safe because preprocessing already crops well past the chin
    into the shoulders/chest, so the trimmed sliver is disposable padding, not
    subject detail. Requires `mesh` to already be oriented with +Z up.
    """
    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    height = z_max - z_min
    if height <= 0:
        raise PrintabilityError("Mesh has zero height after orientation; cannot cut a base.")

    cut_z = z_min + trim_fraction * height
    sliced = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, cut_z], cap=True
    )
    if sliced is None or sliced.vertices.shape[0] == 0:
        raise PrintabilityError("Slicing the base plane produced an empty mesh.")
    return _keep_largest_component(sliced)


def scale_to_height(mesh: trimesh.Trimesh, target_height_mm: float) -> trimesh.Trimesh:
    """Uniformly scale so the mesh's own (Z-axis) extent equals `target_height_mm`.
    Assumes the mesh is already oriented with +Z up."""
    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    current_height = z_max - z_min
    if current_height <= 0:
        raise PrintabilityError("Mesh has zero height; cannot scale to a target height.")
    mesh = mesh.copy()
    mesh.apply_scale(target_height_mm / current_height)
    return mesh


def add_pedestal(
    mesh: trimesh.Trimesh,
    *,
    pedestal_height_mm: float,
    radius_ratio: float,
    overlap_mm: float,
) -> trimesh.Trimesh:
    """Fuse a flat cylindrical pedestal beneath the bust for print-bed stability.

    The pedestal's radius is `radius_ratio` times the bust's own base radius (>1,
    so the pedestal is visibly wider and the print doesn't topple), and it's
    positioned to overlap `overlap_mm` up into the bust's volume — not just touch
    it at a coincident face — so the boolean union (via the `manifold3d` engine) is
    a well-defined solid-solid union rather than a degenerate coplanar case.
    Assumes `mesh` is already oriented (+Z up), base-capped, and scaled to mm.
    """
    z_min = mesh.bounds[0][2]
    footprint = mesh.vertices[mesh.vertices[:, 2] <= z_min + 1e-6]
    if footprint.shape[0] == 0:
        # Numerical edge case: nothing exactly at z_min; use a thin slab instead.
        footprint = mesh.vertices[mesh.vertices[:, 2] <= z_min + 1e-3]
    center_xy = footprint[:, :2].mean(axis=0)
    base_radius = float(np.max(np.linalg.norm(footprint[:, :2] - center_xy, axis=1)))
    if base_radius <= 0:
        raise PrintabilityError("Could not determine the bust's base footprint for the pedestal.")

    pedestal_radius = base_radius * radius_ratio
    pedestal = trimesh.creation.cylinder(
        radius=pedestal_radius, height=pedestal_height_mm + overlap_mm, sections=64
    )
    # cylinder is centered at the origin; move its top face to `overlap_mm` above
    # the bust's base plane (i.e. push it up into the bust) and its footprint under
    # the bust's centroid.
    pedestal.apply_translation(
        [center_xy[0], center_xy[1], z_min - pedestal_height_mm / 2 + overlap_mm / 2]
    )

    fused = trimesh.boolean.union([mesh, pedestal], engine="manifold")
    if fused.vertices.shape[0] == 0:
        raise PrintabilityError("Boolean union of bust and pedestal produced an empty mesh.")
    return fused


# --------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------


def make_printable(
    mesh: trimesh.Trimesh,
    *,
    height_mm: float | None = None,
) -> tuple[trimesh.Trimesh, PrintabilityReport]:
    """Full printability pipeline: repair -> orient -> close base -> scale ->
    pedestal -> final repair pass. Returns the finished mesh and a report."""
    target_height = settings.default_height_mm if height_mm is None else height_mm
    if target_height <= 0:
        raise PrintabilityError(f"height_mm must be positive, got {target_height}")

    mesh = repair_mesh(mesh)
    mesh = orient_upright(mesh)
    mesh = close_base(mesh, settings.base_trim_fraction)
    mesh = scale_to_height(mesh, target_height)
    bust_height_mm = float(mesh.bounds[1][2] - mesh.bounds[0][2])

    mesh = add_pedestal(
        mesh,
        pedestal_height_mm=settings.pedestal_height_mm,
        radius_ratio=settings.pedestal_radius_ratio,
        overlap_mm=settings.pedestal_overlap_mm,
    )

    # Final cleanup pass: booleans can introduce slivers/duplicate geometry at the
    # seam even when the result is technically watertight.
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.fix_normals()
    mesh = _keep_largest_component(mesh)

    # Sit the finished (bust + pedestal) print on the bed at z=0.
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    total_height_mm = float(mesh.bounds[1][2] - mesh.bounds[0][2])
    report = PrintabilityReport(
        bust_height_mm=bust_height_mm,
        total_height_mm=total_height_mm,
        is_watertight=bool(mesh.is_watertight),
        is_winding_consistent=bool(mesh.is_winding_consistent),
        volume_mm3=float(mesh.volume) if mesh.is_watertight else float("nan"),
        face_count=int(mesh.faces.shape[0]),
        vertex_count=int(mesh.vertices.shape[0]),
    )
    return mesh, report
