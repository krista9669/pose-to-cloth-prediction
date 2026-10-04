import argparse
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from garment_config import (
    add_dataset_argument,
    add_garment_argument,
    garment_paths,
    iter_garments,
    load_faces,
    print_supported_garments,
)


# ============================================================
# CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(description="Visualize garment prediction surfaces.")
add_garment_argument(parser)
add_dataset_argument(parser)
args = parser.parse_args()
print_supported_garments()
garments = iter_garments(args.garment)
if len(garments) > 1:
    for garment in garments:
        print(f"\n=== Visualizing garment: {garment} ===")
        command = [sys.executable, __file__, "--garment", garment]
        if args.dataset_dir is not None:
            command.extend(("--dataset-dir", str(args.dataset_dir)))
        subprocess.run(command, check=True)
    raise SystemExit(0)

paths = garment_paths(garments[0])
print(f"Selected garment: {paths.garment}")
RESULTS_DIR = paths.results_dir

SAMPLE_INDEX = 0


# ============================================================
# LOAD PREDICTION AND GROUND TRUTH
# ============================================================

predicted = np.load(
    RESULTS_DIR / "predicted_vertices.npy"
)

ground_truth = np.load(
    RESULTS_DIR / "ground_truth_vertices.npy"
)

print("=" * 60)
print("3D CLOTH SURFACE VISUALIZATION")
print("=" * 60)

print(f"\nPredicted shape:    {predicted.shape}")
print(f"Ground truth shape: {ground_truth.shape}")


pred = predicted[SAMPLE_INDEX]
true = ground_truth[SAMPLE_INDEX]


# ============================================================
# LOAD FACES
# ============================================================

print("\nLoading mesh topology...")

faces = load_faces(paths, args.dataset_dir)
if faces.size and faces.max() >= predicted.shape[1]:
    raise ValueError("Mesh topology contains an index outside the predicted vertex array")

print(f"Faces: {faces.shape}")


# ============================================================
# CREATE TRIANGULAR SURFACES
# ============================================================

true_triangles = true[faces]

pred_triangles = pred[faces]


# ============================================================
# CREATE FIGURE
# ============================================================

fig = plt.figure(figsize=(16, 8))


# ============================================================
# GROUND TRUTH
# ============================================================

ax1 = fig.add_subplot(
    121,
    projection="3d"
)

mesh1 = Poly3DCollection(
    true_triangles,
    alpha=0.85
)

mesh1.set_edgecolor("black")
mesh1.set_linewidth(0.05)

ax1.add_collection3d(mesh1)

ax1.set_title(
    "Ground Truth Cloth",
    fontsize=16
)

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")


# ============================================================
# PREDICTED
# ============================================================

ax2 = fig.add_subplot(
    122,
    projection="3d"
)

mesh2 = Poly3DCollection(
    pred_triangles,
    alpha=0.85
)

mesh2.set_edgecolor("black")
mesh2.set_linewidth(0.05)

ax2.add_collection3d(mesh2)

ax2.set_title(
    "Predicted Cloth",
    fontsize=16
)

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")


# ============================================================
# SAME AXIS LIMITS
# ============================================================

all_points = np.concatenate(
    [true, pred],
    axis=0
)

x_min = all_points[:, 0].min()
x_max = all_points[:, 0].max()

y_min = all_points[:, 1].min()
y_max = all_points[:, 1].max()

z_min = all_points[:, 2].min()
z_max = all_points[:, 2].max()


ax1.set_xlim(x_min, x_max)
ax2.set_xlim(x_min, x_max)

ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

ax1.set_zlim(z_min, z_max)
ax2.set_zlim(z_min, z_max)


# ============================================================
# SAME VIEW ANGLE
# ============================================================

ax1.view_init(
    elev=20,
    azim=-60
)

ax2.view_init(
    elev=20,
    azim=-60
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    RESULTS_DIR / "cloth_surface_comparison.png"
)

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=250,
    bbox_inches="tight"
)

print("\nSaved:")
print(output_path)


# ============================================================
# DISPLAY
# ============================================================

plt.show()