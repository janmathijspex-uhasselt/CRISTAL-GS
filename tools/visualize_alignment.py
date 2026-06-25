#!/usr/bin/env python3
"""
visualize_alignment.py  –  Check PCL ↔ GS alignment interactively.

Usage:
    python tools/visualize_alignment.py \
        --pcl  /path/to/cloud.e57  \
        --gs   /path/to/gaussian_splat/Export_PLY

Paste your pcl2gs matrix into the TRANSFORM block below, then run.
Open3D viewer controls:
    Left-drag   rotate
    Ctrl+drag   pan
    Scroll      zoom
    Q / ESC     quit
"""

import argparse
import os
import glob
import sys
import numpy as np

# ── Optional: install missing deps automatically ──────────────────────────────
try:
    import open3d as o3d
except ImportError:
    print("open3d not found – install with:  pip install open3d")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  PASTE YOUR pcl2gs MATRIX HERE
#  (4×4, maps a PCL point to GS space)
#  Leave as Identity if you just want to see the raw overlap.
# ─────────────────────────────────────────────────────────────────────────────
# PCL2GS = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1],
# ], dtype=np.float64)
PCL2GS = np.array([
    [-0.495275,   -0.868723,  0.00474662,    0.913476],
    [0.868714,   -0.495294, -0.00440392,    0.888633],
    [0.00617643,  0.00194253,    0.999979,    0.635642],
    [0, 0, 0, 1],
], dtype=np.float64)

# Optional uniform scale factor applied to the PCL cloud BEFORE the matrix
# (set to 0.001 for mm→m, 1.0 if already in metres)
PCL_SCALE = 1.0

# Voxel size used to downsample for display (metres). Increase for speed.
DISPLAY_VOXEL = 0.1
# ─────────────────────────────────────────────────────────────────────────────


def find_best_gs_ply(gs_folder: str) -> str:
    """Return the highest-iteration point_cloud.ply inside gs_folder/point_cloud/."""
    pc_dir = os.path.join(gs_folder, "point_cloud")
    best_iter = -1
    best_ply = None
    for d in os.listdir(pc_dir):
        if not d.startswith("iteration_"):
            continue
        ply = os.path.join(pc_dir, d, "point_cloud.ply")
        if os.path.isfile(ply):
            it = int(d.split("_")[1])
            if it > best_iter:
                best_iter = it
                best_ply = ply
    if best_ply is None:
        raise FileNotFoundError(f"No point_cloud.ply found under {pc_dir}")
    print(f"[GS]  Using {best_ply}  (iteration {best_iter})")
    return best_ply


def load_e57(path: str) -> o3d.geometry.PointCloud:
    """Load an E57 file using pye57 (pip install pye57)."""
    try:
        import pye57
    except ImportError:
        raise RuntimeError("pye57 is required for .e57 files:  pip install pye57")

    e57 = pye57.E57(path)
    all_xyz = []
    all_rgb = []
    for i in range(e57.scan_count):
        data = e57.read_scan(i, ignore_missing_fields=True)
        x = data.get("cartesianX", np.array([], dtype=np.float32))
        y = data.get("cartesianY", np.array([], dtype=np.float32))
        z = data.get("cartesianZ", np.array([], dtype=np.float32))
        if len(x) == 0:
            continue
        xyz = np.stack([x, y, z], axis=1).astype(np.float64)
        all_xyz.append(xyz)
        r = data.get("colorRed",   np.full(len(x), 128, dtype=np.uint8))
        g = data.get("colorGreen", np.full(len(x), 128, dtype=np.uint8))
        b = data.get("colorBlue",  np.full(len(x), 128, dtype=np.uint8))
        rgb = np.stack([r, g, b], axis=1).astype(np.float64) / 255.0
        all_rgb.append(rgb)

    if not all_xyz:
        raise RuntimeError(f"No cartesian data found in {path}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_xyz))
    if all_rgb:
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_rgb))
    return pcd


def load_pcl(path: str) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".e57":
        return load_e57(path)
    elif ext in (".ply", ".pcd", ".las", ".laz", ".xyz", ".pts"):
        return o3d.io.read_point_cloud(path)
    else:
        raise ValueError(f"Unsupported PCL format: {ext}")


def bbox_str(pcd: o3d.geometry.PointCloud) -> str:
    pts = np.asarray(pcd.points)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (f"x=[{lo[0]:.3f}, {hi[0]:.3f}]  "
            f"y=[{lo[1]:.3f}, {hi[1]:.3f}]  "
            f"z=[{lo[2]:.3f}, {hi[2]:.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Visualise PCL ↔ GS alignment")
    parser.add_argument("--pcl", required=True, help="Path to PCL file (.e57 / .ply / …)")
    parser.add_argument("--gs",  required=True, help="Path to GS output folder")
    parser.add_argument("--scale", type=float, default=None,
                        help="Override PCL_SCALE at the top of this script")
    parser.add_argument("--voxel", type=float, default=None,
                        help="Override DISPLAY_VOXEL at the top of this script")
    args = parser.parse_args()

    scale  = args.scale  if args.scale  is not None else PCL_SCALE
    voxel  = args.voxel  if args.voxel  is not None else DISPLAY_VOXEL
    matrix = PCL2GS

    # ── Load clouds ───────────────────────────────────────────────────────────
    print(f"\n[PCL] Loading {args.pcl} …")
    pcl = load_pcl(args.pcl)
    print(f"[PCL] {len(pcl.points):,} points   bbox: {bbox_str(pcl)}")

    gs_ply = find_best_gs_ply(args.gs)
    print(f"\n[GS]  Loading {gs_ply} …")
    gs = o3d.io.read_point_cloud(gs_ply)
    print(f"[GS]  {len(gs.points):,} points   bbox: {bbox_str(gs)}")

    # ── Apply optional scale to PCL ───────────────────────────────────────────
    if scale != 1.0:
        print(f"\n[PCL] Applying scale factor {scale}")
        pts = np.asarray(pcl.points) * scale
        pcl.points = o3d.utility.Vector3dVector(pts)
        print(f"[PCL] After scale – bbox: {bbox_str(pcl)}")

    # ── Apply pcl2gs transform ────────────────────────────────────────────────
    is_identity = np.allclose(matrix, np.eye(4))
    if not is_identity:
        print(f"\n[PCL] Applying PCL2GS transform:\n{matrix}")
        pcl.transform(matrix)
        print(f"[PCL] After transform – bbox: {bbox_str(pcl)}")
    else:
        print("\n[PCL] PCL2GS is identity – showing raw overlap.")

    # ── Downsample for display ────────────────────────────────────────────────
    print(f"\n[Vis] Downsampling to voxel={voxel} m …")
    pcl_ds = pcl.voxel_down_sample(voxel)
    gs_ds  = gs.voxel_down_sample(voxel)
    print(f"[Vis] PCL: {len(pcl_ds.points):,} pts   GS: {len(gs_ds.points):,} pts")

    # ── Colour coding: PCL=blue  GS=orange ───────────────────────────────────
    pcl_ds.paint_uniform_color([0.1, 0.4, 1.0])   # blue
    gs_ds.paint_uniform_color( [1.0, 0.6, 0.1])   # orange

    # ── Visualise ─────────────────────────────────────────────────────────────
    print("\n[Vis] Opening viewer  (blue = PCL transformed,  orange = GS)")
    print("      Q / ESC to quit\n")
    o3d.visualization.draw_geometries(
        [pcl_ds, gs_ds],
        window_name="PCL (blue) vs GS (orange)",
        width=1280, height=720,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
