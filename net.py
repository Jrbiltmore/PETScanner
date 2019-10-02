import torch
from torch.nn import functional as F
from torch import nn
import numpy as np


def upscale2d(x, factor=2):
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor, factor)


class Blur(nn.Module):
    def __init__(self, channels):
        super(Blur, self).__init__()
        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :]
        f /= np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)


class FromGrayScale(nn.Module):
    def __init__(self, channels, outputs):
        super(FromGrayScale, self).__init__()
        self.from_grayscale = nn.Conv2d(channels, outputs, 1, 1, 0)

    def forward(self, x):
        x = self.from_grayscale(x)
        x = F.relu(x)

        return x


class Block(nn.Module):
    def __init__(self, inputs, outputs, last=False):
        super(Block, self).__init__()
        self.conv_1 = nn.Conv2d(inputs, inputs, 3, 1, 1, bias=False)
        self.blur = Blur(inputs)
        self.last = last
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()
        if last:
            self.dense = nn.Linear(inputs * 4 * 4, outputs)
        else:
            self.conv_2 = nn.Conv2d(inputs, outputs, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inputs)
        self.bn2 = nn.BatchNorm2d(outputs)

    def forward(self, x):
        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(self.bn1(x), 0.2)

        if self.last:
            x = self.dense(x.view(x.shape[0], -1))
        else:
            x = self.conv_2(self.blur(x))
            x = downscale2d(x)
            x = x + self.bias_2
            x = self.bn2(x)
        x = F.relu(x)
        return x


class StyleGANInspiredNet(torch.nn.Module):
    def __init__(self):
        super(StyleGANInspiredNet, self).__init__()

        inputs = 64
        self.from_grayscale = FromGrayScale(1, inputs)
        self.encode_block: nn.ModuleList[Block] = nn.ModuleList()

        self.encode_block.append(Block(inputs, 2 * inputs, False))

        self.encode_block.append(Block(2 * inputs, 4 * inputs, False))

        self.encode_block.append(Block(4 * inputs, 4 * inputs, True))

        self.fc2 = nn.Linear(4 * inputs, 3)

        self.y_mean = torch.tensor(np.asarray([3.48624905e+01, - 7.88549283e-01,  1.00120885e-03]), dtype=torch.float32)
        self.y_std = torch.tensor(np.asarray([1.71504586, 4.66643101, 1.02884424]), dtype=torch.float32)

    def forward(self, x):
        x -= 10.86
        x /= 27.45
        x = self.from_grayscale(x[:, None])
        x = F.leaky_relu(x, 0.2)

        for i in range(len(self.encode_block)):
            x = self.encode_block[i](x)

        x = self.fc2(x).squeeze()

        return x * self.y_std[None, :] + self.y_mean[None, :]
