"""
Based on https://github.com/MGH-LEMoN/ddpm-labels/blob/main/ddpm_labels/models/model1.py without time embeddings
"""
import math

import torch
from torch import nn
from torchinfo import summary


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, up=False):
        super().__init__()
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(
        self,
        x,
    ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self, image_channels):
        super().__init__()
        image_channels = image_channels
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1])
                for i in range(len(down_channels) - 1)
            ]
        )
        # Upsample
        self.ups = nn.ModuleList(
            [
                Block(up_channels[i], up_channels[i + 1], up=True)
                for i in range(len(up_channels) - 1)
            ]
        )

        self.output = nn.Conv2d(up_channels[-1], image_channels, out_dim)

    def forward(self, x, timestep=None):
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x)#, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x)#, t)
        return self.output(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_channels = 1
    image_size = (160, 192)  # (28, 28)
    batch_size = 1024 + 256 + 128 + 64 + 32

    # Note: For (28, 28), remove 2 up/down channels.

    model = SimpleUnet(image_channels=image_channels).to(device)
    summary(
        model,
        input_size=[(batch_size, image_channels, *image_size), (batch_size,)],
        col_names=["input_size", "output_size", "num_params"],
        depth=5,
    )