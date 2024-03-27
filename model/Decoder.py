"""
Tiny Decoder for extracting message from image
"""
import torch
import torch.nn as nn


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    """Convert tensor into message (binary tensor)"""
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.mask = lambda x : (x > self.threshold).float()

    def forward(self, x):
        x = torch.sigmoid(x)
        return self.mask(x)


class Block(nn.Module):
    """ResBlock2D"""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clamp = Clamp(threshold=0.5)
        self.down = nn.Sequential(
            conv(3, 64), Block(64, 64),
            conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
            conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
            conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
            conv(64, 3)
        )
        

    def forward(self, x):
        x = self.down(x)
        return self.clamp(x)


if __name__ == "__main__":
    """sanity check"""
    model = Decoder()

    tensor = torch.randn(2, 3, 512, 512)

    out = model(tensor)

    print(out)
