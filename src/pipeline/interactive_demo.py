import argparse
import subprocess
import sys

import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from garment_config import (
    add_dataset_argument,
    add_garment_argument,
    garment_paths,
    iter_garments,
    load_faces,
    print_supported_garments,
)
from model import ClothPredictor


# ============================================================
# CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(description="Compare garment predictions interactively.")
add_garment_argument(parser)
add_dataset_argument(parser)
args = parser.parse_args()
print_supported_garments()
garments = iter_garments(args.garment)
if len(garments) > 1:
    for garment in garments:
        print(f"\n=== Running garment: {garment} ===")
        command = [sys.executable, __file__, "--garment", garment]
        if args.dataset_dir is not None:
            command.extend(("--dataset-dir", str(args.dataset_dir)))
        subprocess.run(command, check=True)
    raise SystemExit(0)

paths = garment_paths(args.garment)
print(f"Selected garment: {paths.garment}")
DATA_DIR = paths.data_dir
MODEL_PATH = paths.model_path

START_FRAME = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("INTERACTIVE DATA-DRIVEN CLOTH DEMO")
print("=" * 60)

print(f"\nDevice: {device}")


# ============================================================
# LOAD TEST POSES
# ============================================================

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

Y_test = np.load(
    DATA_DIR / "Y_test.npy"
)

print(
    f"\nTest poses: {X_test.shape}"
)


# ============================================================
# LOAD PCA
# ============================================================

pca = joblib.load(
    DATA_DIR / "pca.pkl"
)
INPUT_DIM = X_test.shape[1]
PCA_DIM = pca.n_components_

print(
    f"PCA components: {pca.n_components_}"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = ClothPredictor(
    input_dim=INPUT_DIM,
    output_dim=PCA_DIM
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD MESH TOPOLOGY
# ============================================================

faces = load_faces(paths, args.dataset_dir)

print(
    f"Mesh faces: {faces.shape}"
)


# ============================================================
# PRECOMPUTE PREDICTIONS
# ============================================================
#
# We predict all test poses once.
#
# This makes the slider extremely responsive.
# ============================================================

print("\nGenerating predictions...")

X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    predicted_pca = model(
        X_tensor
    ).cpu().numpy()


predicted_flat = pca.inverse_transform(
    predicted_pca
)

if Y_test.shape[1] % 3:
    raise ValueError(f"Y_test dimension is not divisible by 3: {Y_test.shape}")
num_vertices = Y_test.shape[1] // 3
if faces.size and (faces.min() < 0 or faces.max() >= num_vertices):
    raise ValueError("Mesh topology contains an index outside the test vertex array")

predicted_vertices = predicted_flat.reshape(
    -1,
    num_vertices,
    3
)

ground_truth_vertices = Y_test.reshape(
    -1,
    num_vertices,
    3
)

print(
    f"Predicted cloth shape: "
    f"{predicted_vertices.shape}"
)

print("\nPredictions ready.")


# ============================================================
# INITIAL FRAME
# ============================================================

current_frame = START_FRAME

pred = predicted_vertices[current_frame]
true = ground_truth_vertices[current_frame]


# ============================================================
# CREATE FIGURE
# ============================================================

fig = plt.figure(
    figsize=(16, 9)
)

fig.suptitle(
    "Data-Driven Cloth for Interactive Applications",
    fontsize=18
)


# ============================================================
# AXES
# ============================================================

ax_true = fig.add_subplot(
    121,
    projection="3d"
)

ax_pred = fig.add_subplot(
    122,
    projection="3d"
)


# ============================================================
# FUNCTION TO GET AXIS LIMITS
# ============================================================

def set_limits(ax, true_vertices, pred_vertices):

    points = np.concatenate(
        [
            true_vertices,
            pred_vertices
        ],
        axis=0
    )

    x_min = points[:, 0].min()
    x_max = points[:, 0].max()

    y_min = points[:, 1].min()
    y_max = points[:, 1].max()

    z_min = points[:, 2].min()
    z_max = points[:, 2].max()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)


# ============================================================
# DRAW MESH
# ============================================================

def draw_mesh(ax, vertices, title):

    triangles = vertices[faces]

    mesh = Poly3DCollection(
        triangles,
        alpha=0.85
    )

    mesh.set_edgecolor("black")
    mesh.set_linewidth(0.05)

    ax.add_collection3d(mesh)

    ax.set_title(
        title,
        fontsize=15
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return mesh


# ============================================================
# INITIAL MESHES
# ============================================================

true_mesh = draw_mesh(
    ax_true,
    true,
    "Ground Truth"
)

pred_mesh = draw_mesh(
    ax_pred,
    pred,
    "Neural Network Prediction"
)


set_limits(
    ax_true,
    true,
    pred
)

set_limits(
    ax_pred,
    true,
    pred
)


# ============================================================
# SAME CAMERA
# ============================================================

ax_true.view_init(
    elev=20,
    azim=-60
)

ax_pred.view_init(
    elev=20,
    azim=-60
)


# ============================================================
# ERROR DISPLAY
# ============================================================

def calculate_error(predicted, actual):

    difference = (
        predicted - actual
    )

    mse = np.mean(
        difference ** 2
    )

    rmse = np.sqrt(mse)

    vertex_error = np.linalg.norm(
        difference,
        axis=1
    )

    mean_error = np.mean(
        vertex_error
    )

    return mse, rmse, mean_error


mse, rmse, mean_error = calculate_error(
    pred,
    true
)


error_text = fig.text(
    0.5,
    0.91,
    (
        f"Frame: {current_frame}    |    "
        f"MSE: {mse:.6f}    |    "
        f"RMSE: {rmse:.6f}    |    "
        f"Mean Vertex Error: {mean_error:.6f}"
    ),
    ha="center",
    fontsize=11
)


# ============================================================
# SLIDER AREA
# ============================================================

slider_axis = plt.axes(
    [0.20, 0.06, 0.60, 0.035]
)

frame_slider = Slider(
    slider_axis,
    "Test Pose",
    0,
    len(X_test) - 1,
    valinit=START_FRAME,
    valstep=1
)


# ============================================================
# SLIDER CALLBACK
# ============================================================

def update(frame):

    frame = int(frame)

    new_pred = predicted_vertices[frame]
    new_true = ground_truth_vertices[frame]

    # Update predicted mesh
    pred_mesh.set_verts(
        new_pred[faces]
    )

    # Update ground truth mesh
    true_mesh.set_verts(
        new_true[faces]
    )

    # Update limits
    set_limits(
        ax_true,
        new_true,
        new_pred
    )

    set_limits(
        ax_pred,
        new_true,
        new_pred
    )

    # Calculate metrics
    mse, rmse, mean_error = calculate_error(
        new_pred,
        new_true
    )

    error_text.set_text(
        (
            f"Frame: {frame}    |    "
            f"MSE: {mse:.6f}    |    "
            f"RMSE: {rmse:.6f}    |    "
            f"Mean Vertex Error: {mean_error:.6f}"
        )
    )

    fig.canvas.draw_idle()


frame_slider.on_changed(update)


# ============================================================
# DISPLAY
# ============================================================

plt.show()