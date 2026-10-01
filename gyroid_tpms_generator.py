#!/usr/bin/env python3
"""
gyroid_tpms_generator.py
========================
Generates a watertight Gyroid TPMS (Triply Periodic Minimal Surface) lattice
and exports it as a binary STL file ready for FDM slicing (Cura, PrusaSlicer).

Dependencies (install via pip):
    pip install numpy scipy scikit-image trimesh

Usage examples:
    python gyroid_tpms_generator.py                     # defaults
    python gyroid_tpms_generator.py --cells 3 3 3 \
        --cell-size 5.0 --t 0.5 --resolution 80 \
        --mode sheet --output gyroid.stl --decimate 0.0
"""

import argparse
import sys
import numpy as np
from skimage import measure
import trimesh

# ──────────────────────────────────────────────────────────────────────────────
# 1. GYROID IMPLICIT FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def gyroid_scalar_field(X, Y, Z):
    """
    Evaluates the gyroid implicit function on a meshgrid.

    Standard trigonometric approximation:
        F(x,y,z) = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)

    The isosurface F = 0 is the zero-mean-curvature Gyroid minimal surface.
    Shifting the isovalue to t > 0 offsets the surface, thickening the solid.

    Parameters
    ----------
    X, Y, Z : ndarray
        Scaled coordinate grids where one unit = 2π corresponds to one unit cell.

    Returns
    -------
    ndarray : scalar field values with same shape as X
    """
    return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)


# ──────────────────────────────────────────────────────────────────────────────
# 2. BUILD THE VOXEL GRID
# ──────────────────────────────────────────────────────────────────────────────
def build_voxel_grid(n_cells=(3, 3, 3),
                     cell_size=5.0,
                     resolution=80,
                     t=0.0,
                     mode="sheet"):
    """
    Creates the scalar volume that will be iso-surface extracted.

    The coordinate mapping is:
        θ_i = 2π * x_i / cell_size     (x_i in physical mm)

    So one period (unit cell) spans exactly `cell_size` mm.

    Sheet ("double") mode  – isovalue band |F| ≤ t
        The solid domain is {|F(x)| ≤ t}.  Extracted as two offset isosurfaces
        at levels +t and -t, then capped to form a closed volume.
        Encoded here as: vol = t - |F|   →  extract at level 0.

    Solid/skeletal mode    – one side of the surface
        Solid domain is {F(x) ≤ t}.
        Encoded as: vol = t - F          →  extract at level 0.

    Parameters
    ----------
    n_cells     : tuple(int,int,int)  number of unit cells in x,y,z
    cell_size   : float               size of one unit cell in mm
    resolution  : int                 voxels per unit cell (governs mesh quality)
    t           : float               isovalue offset controlling wall thickness
                                      / volume fraction.
                                      Typical range: 0.0 (50 % VF) to ~1.5
                                      (higher t → thicker walls, higher density)
    mode        : "sheet" | "solid"   topology type

    Returns
    -------
    vol       : 3-D ndarray of shape (Nx,Ny,Nz)
    spacing   : tuple(float,float,float)  voxel size in mm
    """
    Nx = n_cells[0] * resolution
    Ny = n_cells[1] * resolution
    Nz = n_cells[2] * resolution

    # Physical extents (mm)
    Lx = n_cells[0] * cell_size
    Ly = n_cells[1] * cell_size
    Lz = n_cells[2] * cell_size

    # Build coordinate arrays (closed interval – include both endpoints)
    x = np.linspace(0, 2 * np.pi * n_cells[0], Nx, endpoint=False)
    y = np.linspace(0, 2 * np.pi * n_cells[1], Ny, endpoint=False)
    z = np.linspace(0, 2 * np.pi * n_cells[2], Nz, endpoint=False)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    F = gyroid_scalar_field(X, Y, Z)

    if mode == "sheet":
        # Sheet / double-gyroid: solid where |F| ≤ t
        # → encode as (t - |F|), extract isosurface at 0
        vol = t - np.abs(F)
    else:
        # Solid/skeletal: solid where F ≤ t  (one network)
        # → encode as (t - F), extract isosurface at 0
        vol = t - F

    spacing = (Lx / Nx, Ly / Ny, Lz / Nz)
    return vol, spacing


# ──────────────────────────────────────────────────────────────────────────────
# 3. ISOSURFACE EXTRACTION – MARCHING CUBES
# ──────────────────────────────────────────────────────────────────────────────
def extract_isosurface(vol, spacing, step_size=1):
    """
    Runs scikit-image Marching Cubes (Lewiner variant) to extract the
    zero-isosurface of `vol`.

    The Lewiner variant resolves topological ambiguities and guarantees
    a topologically consistent mesh, avoiding many non-manifold issues
    that arise with the original Lorensen & Cline algorithm.

    Parameters
    ----------
    vol       : ndarray   scalar volume
    spacing   : tuple     physical voxel size in mm (sets real-world scale)
    step_size : int       step size in voxels – 1 = finest, 2 = 2× faster

    Returns
    -------
    vertices : (N,3) float array   mesh vertices in mm
    faces    : (M,3) int array     triangular faces (vertex indices)
    """
    verts, faces, normals, values = measure.marching_cubes(
        vol,
        level=0.0,
        spacing=spacing,       # ← maps voxel indices to mm coordinates
        step_size=step_size,
        allow_degenerate=False,  # remove zero-area triangles
        method="lewiner",
    )
    return verts, faces


# ──────────────────────────────────────────────────────────────────────────────
# 4. MESH POST-PROCESSING WITH TRIMESH
# ──────────────────────────────────────────────────────────────────────────────
def postprocess_mesh(vertices, faces, decimate_ratio=0.0):
    """
    Creates a Trimesh object and repairs / optimises it for FDM printing.

    Steps
    -----
    1. Merge duplicate vertices
    2. Remove degenerate (zero-area) triangles
    3. Fix winding and face normals
    4. Fill small holes (if mesh is not yet watertight)
    5. Optional quadric decimation to reduce file size

    Parameters
    ----------
    vertices      : (N,3) array
    faces         : (M,3) array
    decimate_ratio: float in [0,1)
                    Fraction of faces to *remove*.
                    0.0 → no decimation  |  0.5 → halve face count.

    Returns
    -------
    mesh : trimesh.Trimesh   processed, (hopefully) watertight mesh
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    # Merge vertices that are very close together
    mesh.merge_vertices()

    # Remove degenerate triangles
    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)

    # Fix face winding consistency and normals
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)

    # Try to fill any remaining open boundary holes
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

    # Optional decimation (reduces triangle count, shrinks STL file size)
    if 0.0 < decimate_ratio < 1.0:
        target_faces = int(len(mesh.faces) * (1.0 - decimate_ratio))
        target_faces = max(target_faces, 100)  # safety floor
        mesh = mesh.simplify_quadric_decimation(target_faces)
        trimesh.repair.fix_normals(mesh)

    return mesh


# ──────────────────────────────────────────────────────────────────────────────
# 5. EXPORT
# ──────────────────────────────────────────────────────────────────────────────
def export_stl(mesh, path="gyroid.stl"):
    """Export as binary STL (compact; universally supported by slicers)."""
    stl_bytes = trimesh.exchange.stl.export_stl(mesh)
    with open(path, "wb") as f:
        f.write(stl_bytes)
    print(f"  ✓ Exported: {path}  ({len(mesh.faces):,} faces, "
          f"{len(mesh.vertices):,} vertices)")
    print(f"    Watertight: {mesh.is_watertight}")
    print(f"    Bounding box (mm): {np.round(mesh.bounding_box.extents, 2)}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. VOLUME FRACTION HELPER
# ──────────────────────────────────────────────────────────────────────────────
def estimate_volume_fraction(vol):
    """
    Estimates the relative density (volume fraction) of the solid domain.

    For sheet mode the solid domain is where vol ≥ 0 (i.e. |F| ≤ t).
    This gives a fast Monte-Carlo-equivalent estimate from the voxel grid.

    Returns
    -------
    vf : float   approximate volume fraction  (0 = fully open, 1 = fully solid)
    """
    return float(np.mean(vol >= 0))


# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate a watertight Gyroid TPMS lattice STL for FDM printing."
    )
    parser.add_argument("--cells", nargs=3, type=int, default=[3, 3, 3],
                        metavar=("NX", "NY", "NZ"),
                        help="Number of unit cells in X, Y, Z (default: 3 3 3)")
    parser.add_argument("--cell-size", type=float, default=5.0,
                        help="Physical size of one unit cell in mm (default: 5.0)")
    parser.add_argument("--t", type=float, default=0.5,
                        help="Level-set offset t controlling wall thickness. "
                             "Range ≈ 0.0 (thin, ~50 %% VF) to 1.5 (thick, ~80 %% VF). "
                             "Default: 0.5")
    parser.add_argument("--resolution", type=int, default=60,
                        help="Voxels per unit cell (default: 60). "
                             "Higher = finer mesh but more RAM/time.")
    parser.add_argument("--mode", choices=["sheet", "solid"], default="sheet",
                        help="'sheet' = hollow-wall (double-gyroid) topology; "
                             "'solid' = skeletal (one-network) topology. Default: sheet")
    parser.add_argument("--step-size", type=int, default=1,
                        help="Marching cubes step size (1=finest, default: 1)")
    parser.add_argument("--decimate", type=float, default=0.0,
                        help="Fraction of faces to remove via decimation [0,1). "
                             "E.g. 0.5 halves the face count. Default: 0.0")
    parser.add_argument("--output", type=str, default="gyroid.stl",
                        help="Output STL file path (default: gyroid.stl)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Gyroid TPMS Generator")
    print("=" * 60)
    print(f"  Mode          : {args.mode}")
    print(f"  Unit cells    : {args.cells[0]} × {args.cells[1]} × {args.cells[2]}")
    print(f"  Cell size     : {args.cell_size} mm")
    total = [args.cells[i] * args.cell_size for i in range(3)]
    print(f"  Total size    : {total[0]} × {total[1]} × {total[2]} mm")
    print(f"  t (offset)    : {args.t}")
    print(f"  Resolution    : {args.resolution} voxels/cell")
    print(f"  Decimation    : {args.decimate*100:.0f}% face removal")
    print()

    # -- Step 1: Build voxel grid --
    print("[1/4] Building voxel grid …")
    vol, spacing = build_voxel_grid(
        n_cells=tuple(args.cells),
        cell_size=args.cell_size,
        resolution=args.resolution,
        t=args.t,
        mode=args.mode,
    )
    vf = estimate_volume_fraction(vol)
    print(f"      Volume fraction (relative density) ≈ {vf:.1%}")
    print(f"      Grid shape: {vol.shape}  ({vol.nbytes / 1e6:.1f} MB)")

    # -- Step 2: Marching Cubes --
    print("[2/4] Extracting isosurface (Marching Cubes, Lewiner) …")
    verts, faces = extract_isosurface(vol, spacing, step_size=args.step_size)
    print(f"      Raw mesh: {len(verts):,} vertices, {len(faces):,} faces")

    # Free the large volume array now that we have the mesh
    del vol

    # -- Step 3: Post-process --
    print("[3/4] Post-processing (merge vertices, fix winding, fill holes) …")
    mesh = postprocess_mesh(verts, faces, decimate_ratio=args.decimate)
    print(f"      Processed: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.faces):,} faces")

    # -- Step 4: Export --
    print(f"[4/4] Exporting STL → {args.output}")
    export_stl(mesh, path=args.output)
    print()
    print("Done. Import the STL into Cura or PrusaSlicer.")
    print("Tip: use 0% infill in the slicer – the Gyroid IS the structure.")


if __name__ == "__main__":
    main()
