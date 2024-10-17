import torch
import torch.nn as nn
import numpy as np

class ConvBlock(nn.Module):
    """
    Convolutional block with two convolutional layers with batch normalization and ReLU activation
    Input: in_channels (int) - number of input channels
    Output: out_channels (int) - number of output channels
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class EncoderBlock(nn.Module):
    """
    Encoder block with a convolutional block and max pooling
    Input: in_channels (int) - number of input channels
    Output: out_channels (int) - number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, 3)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    """
    Decoder block with a transposed convolutional layer and a convolutional block
    Input: in_channels (int) - number of input channels
    Output: out_channels (int) - number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, 3)

    def forward(self, x, skip_features):
        x = self.transpose(x)
        x = torch.cat([x, skip_features], axis=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    """
    UNet model with a contracting path and an expanding path
    Input: in_channels (int) - number of input channels
    Output: out_channels (int) - number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels, 32, 7, padding=3, stride=1)
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)

        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x = self.convolution1(x)
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        b1 = self.bottleneck(p4)

        d4 = self.decoder4(b1, s4)
        d3 = self.decoder3(d4, s3)
        d2 = self.decoder2(d3, s2)
        d1 = self.decoder1(d2, s1)

        return self.output(d1)

class Backwarp(nn.Module):
    def __init__(self, height, width, device):
        super(Backwarp, self).__init__()
        gridX, gridY = np.meshgrid(np.arange(width), np.arange(height))
        self.height = height
        self.width = width
        self.device = device
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, flow):
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = torch.unsqueeze(self.gridX, 0) + u
        y = torch.unsqueeze(self.gridY, 0) + v

        x = 2 * (x / (self.width - 1)) - 1
        y = 2 * (y / (self.height - 1)) - 1
        grid = torch.stack((x, y), dim=3)
        warped_img = nn.functional.grid_sample(img, grid, align_corners=True)
        return warped_img

t = 0.5

def GetFlowCoefficients(device):
    C00 = -(1 - t) * t
    C01 = t * t
    C10 = (1 - t) * (1 - t)
    C11 = -t * (1 - t)

    return (torch.tensor(C00)[None, None, None, None].to(device),
            torch.tensor(C01)[None, None, None, None].to(device),
            torch.tensor(C10)[None, None, None, None].to(device),
            torch.tensor(C11)[None, None, None, None].to(device))

