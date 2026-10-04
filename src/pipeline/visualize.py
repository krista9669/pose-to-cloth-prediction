import argparse
import numpy as np
import matplotlib.pyplot as plt

from garment_config import add_garment_argument, garment_paths, iter_garments, print_supported_garments


def run_visualize(garment: str) -> None:
    results_dir = garment_paths(garment).results_dir
    predicted = np.load(results_dir / "predicted_vertices.npy")
    ground_truth = np.load(results_dir / "ground_truth_vertices.npy")

    print(f"Selected garment: {garment}")
    print("=" * 60)
    print("CLOTH PREDICTION VISUALIZATION")
    print("=" * 60)
    print(f"\nPredicted shape:    {predicted.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")

    sample_index = 0
    pred = predicted[sample_index]
    true = ground_truth[sample_index]
    print(f"\nVisualizing test sample: {sample_index}")

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    ax1.scatter(true[:, 0], true[:, 1], true[:, 2], s=1)
    ax1.set_title("Ground Truth Cloth")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=1)
    ax2.set_title("Predicted Cloth")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    all_points = np.concatenate([true, pred], axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax2.set_zlim(z_min, z_max)

    output_path = results_dir / "cloth_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print("\nVisualization saved to:")
    print(output_path)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize garment prediction point clouds.")
    add_garment_argument(parser)
    args = parser.parse_args()
    print_supported_garments()
    for garment in iter_garments(args.garment):
        print(f"\n=== Running garment: {garment} ===")
        run_visualize(garment)
