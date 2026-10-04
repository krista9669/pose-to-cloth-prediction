import argparse
import subprocess
import sys

import numpy as np
import torch
import joblib

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

VERTEX_DIM = 3
parser = argparse.ArgumentParser(description="Evaluate a trained garment predictor.")
add_garment_argument(parser)
add_dataset_argument(parser)
parser.add_argument(
    "--split",
    choices=("validation", "test"),
    default="test",
    help="Dataset split to evaluate (default: test).",
)
args = parser.parse_args()
print_supported_garments()
garments = iter_garments(args.garment)
if len(garments) > 1:
    for garment in garments:
        print(f"\n=== Running garment: {garment} ===")
        command = [sys.executable, __file__, "--garment", garment, "--split", args.split]
        if args.dataset_dir is not None:
            command.extend(("--dataset-dir", str(args.dataset_dir)))
        subprocess.run(command, check=True)
    raise SystemExit(0)

paths = garment_paths(args.garment)
print(f"Selected garment: {paths.garment}")
DATA_DIR = paths.data_dir
MODEL_PATH = paths.model_path
RESULTS_DIR = paths.results_dir
DATA_SPLIT = "val" if args.split == "validation" else "test"
OUTPUT_PREFIX = "validation_" if args.split == "validation" else ""
METRICS_FILENAME = "validation_metrics.txt" if args.split == "validation" else "evaluation_metrics.txt"


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print(f"CLOTH PREDICTION {args.split.upper()} EVALUATION")
print("=" * 60)

print(f"\nDevice: {device}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print(f"\nLoading {args.split} data...")

X_eval = np.load(
    DATA_DIR / f"X_{DATA_SPLIT}.npy"
)

Y_eval_pca = np.load(
    DATA_DIR / f"Y_{DATA_SPLIT}_pca.npy"
)

Y_eval = np.load(
    DATA_DIR / f"Y_{DATA_SPLIT}.npy"
)

print(f"X_{DATA_SPLIT} shape:      {X_eval.shape}")
print(f"Y_{DATA_SPLIT} PCA shape:  {Y_eval_pca.shape}")
print(f"Y_{DATA_SPLIT} raw shape:  {Y_eval.shape}")
if Y_eval.shape[1] % VERTEX_DIM:
    raise ValueError(f"Y_{DATA_SPLIT} dimension is not divisible by {VERTEX_DIM}: {Y_eval.shape}")
NUM_VERTICES = Y_eval.shape[1] // VERTEX_DIM


# ============================================================
# LOAD PCA
# ============================================================

print("\nLoading PCA model...")

pca = joblib.load(
    DATA_DIR / "pca.pkl"
)
INPUT_DIM = X_eval.shape[1]
PCA_DIM = pca.n_components_

print(
    f"PCA components: "
    f"{pca.n_components_}"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained neural network...")

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
# CONVERT TEST INPUT TO TENSOR
# ============================================================

X_eval_tensor = torch.tensor(
    X_eval,
    dtype=torch.float32
).to(device)


# ============================================================
# PREDICT PCA COEFFICIENTS
# ============================================================

print("\nGenerating cloth predictions...")

with torch.no_grad():

    predicted_pca = model(
        X_eval_tensor
    ).cpu().numpy()


print(
    f"Predicted PCA shape: "
    f"{predicted_pca.shape}"
)


# ============================================================
# RECONSTRUCT 3D CLOTH
# ============================================================
#
# Neural network:
#
# Pose (72)
#      ↓
# Neural Network
#      ↓
# PCA coefficients
#      ↓
# PCA inverse transform
#      ↓
# Flattened garment coordinates
#      ↓
# vertex_count × 3 coordinates
#
# ============================================================

print("\nReconstructing 3D cloth vertices...")

predicted_vertices_flat = pca.inverse_transform(
    predicted_pca
)

print(
    f"Reconstructed flat shape: "
    f"{predicted_vertices_flat.shape}"
)


# Reshape:
#
# (samples, vertex_count * 3)
#        ↓
# (samples, vertex_count, 3)
predicted_vertices = predicted_vertices_flat.reshape(
    -1,
    NUM_VERTICES,
    VERTEX_DIM
)

ground_truth_vertices = Y_eval.reshape(
    -1,
    NUM_VERTICES,
    VERTEX_DIM
)

print(
    f"Predicted vertices shape: "
    f"{predicted_vertices.shape}"
)

print(
    f"Ground truth shape: "
    f"{ground_truth_vertices.shape}"
)


# ============================================================
# METRIC 1 — COORDINATE MSE
# ============================================================

difference = (
    predicted_vertices
    - ground_truth_vertices
)

mse = np.mean(
    difference ** 2
)


# ============================================================
# METRIC 2 — RMSE
# ============================================================

rmse = np.sqrt(mse)


# ============================================================
# METRIC 3 — MEAN PER-VERTEX ERROR
# ============================================================

vertex_errors = np.linalg.norm(
    difference,
    axis=2
)

mean_vertex_error = np.mean(
    vertex_errors
)


# ============================================================
# METRIC 4 — MAXIMUM VERTEX ERROR
# ============================================================

max_vertex_error = np.max(
    vertex_errors
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print(f"{args.split.upper()} EVALUATION RESULTS")
print("=" * 60)

print(
    f"\nCoordinate MSE:       {mse:.8f}"
)

print(
    f"Coordinate RMSE:      {rmse:.8f}"
)

print(
    f"Mean vertex error:    {mean_vertex_error:.8f}"
)

print(
    f"Maximum vertex error: {max_vertex_error:.8f}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print("\nSaving prediction results...")

np.save(
    RESULTS_DIR / f"{OUTPUT_PREFIX}predicted_vertices.npy",
    predicted_vertices
)

np.save(
    RESULTS_DIR / f"{OUTPUT_PREFIX}ground_truth_vertices.npy",
    ground_truth_vertices
)

np.save(
    RESULTS_DIR / f"{OUTPUT_PREFIX}predicted_pca.npy",
    predicted_pca
)


# ============================================================
# LOAD MESH FACES
# ============================================================

print(f"\nLoading {paths.garment} mesh topology...")
faces = load_faces(paths, args.dataset_dir)
if faces.size and (faces.min() < 0 or faces.max() >= NUM_VERTICES):
    raise ValueError(f"Mesh topology references vertices outside 0..{NUM_VERTICES - 1}")

print(
    f"Faces shape: {faces.shape}"
)


# ============================================================
# OBJ WRITER
# ============================================================

def save_obj(vertices, faces, filename):

    with open(filename, "w") as f:

        # Write vertices
        for vertex in vertices:

            x, y, z = vertex

            f.write(
                f"v {x:.6f} {y:.6f} {z:.6f}\n"
            )

        # Write triangular faces
        for face in faces:

            # OBJ uses 1-based indexing
            i, j, k = face + 1

            f.write(
                f"f {i} {j} {k}\n"
            )


# ============================================================
# SAVE ONE PREDICTED MESH
# ============================================================

sample_index = 0

predicted_obj = RESULTS_DIR / f"{OUTPUT_PREFIX}predicted_cloth.obj"

ground_truth_obj = RESULTS_DIR / f"{OUTPUT_PREFIX}ground_truth_cloth.obj"

save_obj(
    predicted_vertices[sample_index],
    faces,
    predicted_obj
)

save_obj(
    ground_truth_vertices[sample_index],
    faces,
    ground_truth_obj
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_file = RESULTS_DIR / METRICS_FILENAME

with open(metrics_file, "w") as f:

    f.write(
        f"Data-Driven Cloth {args.split.title()} Evaluation\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        f"Number of {args.split} samples: "
        f"{len(X_eval)}\n"
    )

    f.write(
        f"Input dimension: "
        f"{INPUT_DIM}\n"
    )

    f.write(
        f"PCA dimension: "
        f"{PCA_DIM}\n"
    )

    f.write(
        f"Original output dimension: "
        f"{NUM_VERTICES * VERTEX_DIM}\n\n"
    )

    f.write(
        f"Coordinate MSE: "
        f"{mse:.8f}\n"
    )

    f.write(
        f"Coordinate RMSE: "
        f"{rmse:.8f}\n"
    )

    f.write(
        f"Mean vertex error: "
        f"{mean_vertex_error:.8f}\n"
    )

    f.write(
        f"Maximum vertex error: "
        f"{max_vertex_error:.8f}\n"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print(f"{args.split.upper()} EVALUATION COMPLETE")
print("=" * 60)

print("\nResults saved:")
print(RESULTS_DIR / f"{OUTPUT_PREFIX}predicted_vertices.npy")
print(RESULTS_DIR / f"{OUTPUT_PREFIX}ground_truth_vertices.npy")
print(RESULTS_DIR / f"{OUTPUT_PREFIX}predicted_pca.npy")
print(RESULTS_DIR / f"{OUTPUT_PREFIX}predicted_cloth.obj")
print(RESULTS_DIR / f"{OUTPUT_PREFIX}ground_truth_cloth.obj")
print(RESULTS_DIR / METRICS_FILENAME)

print("\nYou can now visualize the predicted and")
print("ground-truth cloth meshes.")