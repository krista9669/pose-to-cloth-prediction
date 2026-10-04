import torch
import torch.nn as nn


class ClothPredictor(nn.Module):

    def __init__(self, input_dim=72, output_dim=64):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":

    model = ClothPredictor()

    print(model)

    # Test forward pass
    sample = torch.randn(32, 72)

    output = model(sample)

    print("\nInput shape:", sample.shape)
    print("Output shape:", output.shape)