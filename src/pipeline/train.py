import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from garment_config import add_garment_argument, garment_paths, iter_garments, print_supported_garments
from model import ClothPredictor


BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001


def run_training(garment: str) -> None:
    paths = garment_paths(garment)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Selected garment: {garment}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    X_train = np.load(paths.data_dir / "X_train.npy")
    Y_train = np.load(paths.data_dir / "Y_train_pca.npy")
    X_val = np.load(paths.data_dir / "X_val.npy")
    Y_val = np.load(paths.data_dir / "Y_val_pca.npy")
    print("Training data:", X_train.shape, Y_train.shape)
    print("Validation data:", X_val.shape, Y_val.shape)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ClothPredictor(input_dim=X_train.shape[1], output_dim=Y_train.shape[1]).to(device)
    print("\nModel:")
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting training...\n")
    history = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, Y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        history.append((epoch + 1, train_loss, val_loss))
        print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), paths.model_path)
    with (paths.results_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_loss", "val_loss"))
        writer.writerows(history)

    print("\nTraining complete!")
    print("Model saved to:")
    print(paths.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the garment pose-to-PCA predictor.")
    add_garment_argument(parser)
    args = parser.parse_args()
    print_supported_garments()
    for garment in iter_garments(args.garment):
        print(f"\n=== Running garment: {garment} ===")
        run_training(garment)
