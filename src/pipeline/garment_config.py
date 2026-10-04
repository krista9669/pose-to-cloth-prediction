import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GARMENTS = ("tshirt", "dress")
ALL_GARMENT = "all"
GARMENT_ALIASES = {
    "t-shirt": "tshirt",
    "t_shirt": "tshirt",
    "tshirt": "tshirt",
    "dress": "dress",
    "all": "all",
}


def normalize_garment(garment: str) -> str:
    normalized = garment.strip().lower().replace("_", "-")
    return GARMENT_ALIASES.get(normalized, normalized)


def iter_garments(garment: str | None = None) -> list[str]:
    requested = normalize_garment(garment or ALL_GARMENT)
    if requested == ALL_GARMENT:
        return list(GARMENTS)
    if requested not in GARMENTS:
        raise ValueError(f"Unsupported garment: {garment}")
    return [requested]


def print_supported_garments() -> None:
    print(f"Supported garments: {', '.join(GARMENTS)}")


@dataclass(frozen=True)
class GarmentPaths:
    garment: str
    data_dir: Path
    model_dir: Path
    results_dir: Path

    @property
    def model_path(self) -> Path:
        return self.model_dir / "cloth_predictor.pth"

    @property
    def faces_path(self) -> Path:
        return self.data_dir / "faces.npy"


def add_garment_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--garment",
        type=normalize_garment,
        choices=(*GARMENTS, ALL_GARMENT),
        default=ALL_GARMENT,
        help="Garment pipeline to use (default: all). Supported: all, tshirt, dress.",
    )


def add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Path to this garment's simulations directory; overrides CLOTH_DATASET_ROOT.",
    )


def garment_paths(garment: str) -> GarmentPaths:
    normalized = normalize_garment(garment)
    if normalized not in GARMENTS:
        raise ValueError(f"Unsupported garment: {garment}")
    return GarmentPaths(
        garment=normalized,
        data_dir=REPO_ROOT / "data" / "processed" / normalized,
        model_dir=REPO_ROOT / "models" / normalized,
        results_dir=REPO_ROOT / "results" / normalized,
    )


def simulations_dir(garment: str, dataset_dir: Path | None = None) -> Path:
    if dataset_dir is not None:
        return dataset_dir.expanduser().resolve()

    dataset_root = os.environ.get("CLOTH_DATASET_ROOT")
    if not dataset_root:
        raise FileNotFoundError(
            "Raw simulations are outside this repository. Pass --dataset-dir "
            "or set CLOTH_DATASET_ROOT to the external dataset root."
        )

    paths = garment_paths(garment)
    return Path(dataset_root).expanduser() / paths.garment / "simulations"


def load_faces(
    paths: GarmentPaths,
    dataset_dir: Path | None = None,
) -> np.ndarray:
    if paths.faces_path.exists():
        return np.load(paths.faces_path).astype(np.int32, copy=False)

    source_dir = simulations_dir(paths.garment, dataset_dir)
    pkl_files = sorted(source_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {source_dir}")

    with pkl_files[0].open("rb") as stream:
        mesh_data = pickle.load(stream)
    faces = np.asarray(mesh_data["faces"], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces, got shape {faces.shape}")
    return faces