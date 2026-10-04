import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib

from garment_config import (
    add_dataset_argument,
    add_garment_argument,
    garment_paths,
    iter_garments,
    print_supported_garments,
    simulations_dir,
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_FILES = 100
N_COMPONENTS = 64
RANDOM_STATE = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare garment pose and vertex data.")
    add_garment_argument(parser)
    add_dataset_argument(parser)
    parser.add_argument(
        "--topology-only",
        action="store_true",
        help="Cache the selected garment topology without rebuilding processed arrays.",
    )
    args = parser.parse_args()
    print_supported_garments()
    garments = iter_garments(args.garment)
    if len(garments) > 1:
        for garment in garments:
            print(f"\n=== Preparing garment: {garment} ===")
            command = [sys.executable, __file__, "--garment", garment]
            if args.dataset_dir is not None:
                source_dir = args.dataset_dir / garment / "simulations"
                command.extend(("--dataset-dir", str(source_dir)))
            if args.topology_only:
                command.append("--topology-only")
            subprocess.run(command, check=True)
        return

    paths = garment_paths(args.garment)
    print(f"Selected garment: {paths.garment}")
    required_outputs = (
        paths.faces_path,
        paths.data_dir / "mesh_metadata.json",
        paths.data_dir / "X_train.npy",
        paths.data_dir / "X_val.npy",
        paths.data_dir / "X_test.npy",
        paths.data_dir / "Y_train_pca.npy",
        paths.data_dir / "Y_val.npy",
        paths.data_dir / "Y_val_pca.npy",
        paths.data_dir / "Y_test.npy",
        paths.data_dir / "Y_test_pca.npy",
        paths.data_dir / "pca.pkl",
    )
    has_raw_dataset_config = args.dataset_dir is not None or os.environ.get("CLOTH_DATASET_ROOT")
    if not has_raw_dataset_config and all(path.exists() for path in required_outputs):
        print(f"Processed dataset already exists at {paths.data_dir}; skipping preparation.")
        print("Configure CLOTH_DATASET_ROOT or pass --dataset-dir to rebuild it from raw simulations.")
        return

    source_dir = simulations_dir(paths.garment, args.dataset_dir)
    pkl_files = sorted(source_dir.glob("*.pkl"))[:NUM_FILES]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {source_dir}")

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Garment: {paths.garment}")
    print(f"Dataset directory: {source_dir}")
    print(f"Simulation files found: {len(pkl_files)}")

    with pkl_files[0].open("rb") as stream:
        first_data = pickle.load(stream)
    first_vertices = np.asarray(first_data["vertices"])
    faces = np.asarray(first_data["faces"], dtype=np.int32)
    if first_vertices.ndim != 3 or first_vertices.shape[2] != 3:
        raise ValueError(f"Unexpected vertices shape: {first_vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces, got {faces.shape}")
    vertex_count = first_vertices.shape[1]
    if faces.size and (faces.min() < 0 or faces.max() >= vertex_count):
        raise ValueError("Mesh faces reference vertices outside the garment mesh")
    np.save(paths.faces_path, faces)
    metadata = {
        "garment": paths.garment,
        "vertex_count": int(vertex_count),
        "vertex_dim": 3,
        "pose_dim": int(np.asarray(first_data["pose"]).shape[-1]),
        "pca_components": N_COMPONENTS,
        "simulation_files": len(pkl_files),
    }
    with (paths.data_dir / "mesh_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    if args.topology_only:
        print(f"Cached {vertex_count} vertices and {len(faces)} faces in {paths.data_dir}")
        return

    with tempfile.TemporaryDirectory(prefix="cloth_prepare_") as staging_dir:
        staging_dir = Path(staging_dir)
        pose_file = staging_dir / "poses.dat"
        vertices_file = staging_dir / "vertices.dat"
        total_frames = 0

        with pose_file.open("wb") as pose_stream, vertices_file.open("wb") as vertices_stream:
            for index, file_path in enumerate(pkl_files, start=1):
                if index == 1:
                    data = first_data
                else:
                    with file_path.open("rb") as stream:
                        data = pickle.load(stream)
                pose = np.asarray(data["pose"], dtype=np.float32)
                vertices = np.asarray(data["vertices"], dtype=np.float32)
                if pose.ndim != 2:
                    raise ValueError(f"Unexpected pose shape in {file_path.name}: {pose.shape}")
                if vertices.ndim != 3 or vertices.shape[1:] != (vertex_count, 3):
                    raise ValueError(f"Inconsistent vertices shape in {file_path.name}: {vertices.shape}")
                if pose.shape[0] != vertices.shape[0]:
                    raise ValueError(
                        f"Frame mismatch in {file_path.name}: pose={pose.shape[0]}, "
                        f"vertices={vertices.shape[0]}"
                    )
                if pose.shape[1] != metadata["pose_dim"]:
                    raise ValueError(f"Inconsistent pose dimension in {file_path.name}: {pose.shape}")
                file_faces = np.asarray(data["faces"], dtype=np.int32)
                if not np.array_equal(file_faces, faces):
                    raise ValueError(f"Inconsistent mesh topology in {file_path.name}")
                pose.tofile(pose_stream)
                vertices.reshape(vertices.shape[0], -1).tofile(vertices_stream)
                total_frames += pose.shape[0]
                print(f"[{index:3d}/{len(pkl_files)}] {file_path.name:40s} frames={pose.shape[0]}")
                del data, pose, vertices

        pose_dim = metadata["pose_dim"]
        coordinate_dim = vertex_count * 3
        X = np.memmap(pose_file, dtype=np.float32, mode="r", shape=(total_frames, pose_dim))
        Y = np.memmap(vertices_file, dtype=np.float32, mode="r", shape=(total_frames, coordinate_dim))
        train_indices, temp_indices = train_test_split(
            np.arange(total_frames), test_size=0.30, random_state=RANDOM_STATE, shuffle=True
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.50, random_state=RANDOM_STATE, shuffle=True
        )

        X_train = np.asarray(X[train_indices])
        X_val = np.asarray(X[val_indices])
        X_test = np.asarray(X[test_indices])
        Y_train = np.asarray(Y[train_indices])
        pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
        Y_train_pca = pca.fit_transform(Y_train)
        del Y_train

        Y_val = np.asarray(Y[val_indices])
        Y_val_pca = pca.transform(Y_val)
        np.save(paths.data_dir / "Y_val.npy", Y_val)
        np.save(paths.data_dir / "Y_val_pca.npy", Y_val_pca)
        del Y_val, Y_val_pca

        Y_test = np.asarray(Y[test_indices])
        Y_test_pca = pca.transform(Y_test)
        np.save(paths.data_dir / "Y_test.npy", Y_test)
        np.save(paths.data_dir / "Y_test_pca.npy", Y_test_pca)
        del Y_test, Y_test_pca

        np.save(paths.data_dir / "X_train.npy", X_train)
        np.save(paths.data_dir / "Y_train_pca.npy", Y_train_pca)
        np.save(paths.data_dir / "X_val.npy", X_val)
        np.save(paths.data_dir / "X_test.npy", X_test)
        joblib.dump(pca, paths.data_dir / "pca.pkl")

        print(f"\nSaved processed data to {paths.data_dir}")
        print(f"Input shape: ({total_frames}, {pose_dim}); vertex shape: ({vertex_count}, 3)")
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum() * 100:.4f}%")
        X._mmap.close()
        Y._mmap.close()


if __name__ == "__main__":
    main()
