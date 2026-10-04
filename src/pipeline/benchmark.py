import argparse
import time

import numpy as np
import torch
import joblib

from garment_config import add_garment_argument, garment_paths, iter_garments, print_supported_garments
from model import ClothPredictor

NUM_RUNS = 100


def run_benchmark(garment: str) -> None:
    paths = garment_paths(garment)
    data_dir = paths.data_dir
    model_path = paths.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Selected garment: {garment}")
    print("=" * 60)
    print("DATA-DRIVEN CLOTH INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"\nDevice: {device}")

    X_test = np.load(data_dir / "X_test.npy")
    sample = X_test[0]
    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)

    pca = joblib.load(data_dir / "pca.pkl")
    input_dim = X_test.shape[1]
    pca_dim = pca.n_components_

    model = ClothPredictor(input_dim=input_dim, output_dim=pca_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print("\nWarming up model...")
    with torch.no_grad():
        for _ in range(20):
            model(sample_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()

    print("\nBenchmarking neural network...")
    nn_times = []
    with torch.no_grad():
        for _ in range(NUM_RUNS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(sample_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            nn_times.append(end - start)

    nn_mean = np.mean(nn_times)
    nn_ms = nn_mean * 1000

    print("Benchmarking PCA reconstruction...")
    pca_times = []
    with torch.no_grad():
        prediction_cpu = model(sample_tensor).detach().cpu().numpy()
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        pca.inverse_transform(prediction_cpu)
        end = time.perf_counter()
        pca_times.append(end - start)

    pca_mean = np.mean(pca_times)
    pca_ms = pca_mean * 1000
    total_ms = nn_ms + pca_ms
    fps = 1000 / total_ms

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nNeural network inference: {nn_ms:.4f} ms")
    print(f"PCA reconstruction:       {pca_ms:.4f} ms")
    print(f"Total prediction time:    {total_ms:.4f} ms")
    print(f"Estimated prediction FPS: {fps:.2f}")

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = paths.results_dir / "inference_benchmark.txt"
    with open(results_path, "w") as f:
        f.write(f"Garment: {garment}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Neural network inference: {nn_ms:.4f} ms\n")
        f.write(f"PCA reconstruction: {pca_ms:.4f} ms\n")
        f.write(f"Total prediction time: {total_ms:.4f} ms\n")
        f.write(f"Estimated prediction FPS: {fps:.2f}\n")

    print(f"\nSaved benchmark to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark garment prediction inference.")
    add_garment_argument(parser)
    args = parser.parse_args()
    print_supported_garments()
    for garment in iter_garments(args.garment):
        print(f"\n=== Running garment: {garment} ===")
        run_benchmark(garment)
