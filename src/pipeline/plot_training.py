import argparse
import csv
import subprocess
import sys
import matplotlib.pyplot as plt

from garment_config import add_garment_argument, garment_paths, iter_garments, print_supported_garments


parser = argparse.ArgumentParser(description="Plot garment training and validation loss.")
add_garment_argument(parser)
args = parser.parse_args()
print_supported_garments()
garments = iter_garments(args.garment)
if len(garments) > 1:
    for garment in garments:
        print(f"\n=== Plotting garment: {garment} ===")
        subprocess.run(
            [sys.executable, __file__, "--garment", garment],
            check=True,
        )
    raise SystemExit(0)

paths = garment_paths(garments[0])
print(f"Selected garment: {paths.garment}")


# ============================================================
# TRAINING RESULTS
# ============================================================

epochs = [
    1,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100
]

train_loss = [
    99.117988,
    22.067582,
    10.563653,
    6.222552,
    4.285701,
    3.869607,
    2.743935,
    2.284440,
    1.527102,
    1.748108,
    0.974768
]

val_loss = [
    73.784012,
    22.998413,
    12.917937,
    7.742776,
    4.890603,
    3.763932,
    3.350342,
    1.960781,
    1.742367,
    1.467753,
    1.303923
]


history_path = paths.results_dir / "training_history.csv"
if history_path.exists():
    with history_path.open("r", newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
else:
    raise FileNotFoundError(f"No training history found at {history_path}; train this garment first.")

paths.results_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# CREATE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_path = paths.results_dir / "training_validation_loss.png"

plt.savefig(
    output_path,
    dpi=250,
    bbox_inches="tight"
)

print("=" * 60)
print("TRAINING CURVE GENERATED")
print("=" * 60)

print(f"\nSaved to:")
print(output_path)

plt.show()