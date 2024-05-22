import torch
import torch.nn as nn
from einops import rearrange
from Vit.vit import Vit


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super(EncoderBottleneck, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )

        width = int(out_channels * (base_width / 64.))

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features=width)

        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(num_features=width)

        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = x + x_down
        x = self.relu(x)
        return x

        pass


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DecoderBottleneck, self).__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat((x_concat, x), dim=1)
        x = self.layer(x)
        return x
        pass


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(in_channels=out_channels, out_channels=out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(in_channels=out_channels * 2, out_channels=out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(in_channels=out_channels * 4, out_channels=out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim

        # self.vit = Vit()
        self.conv2 = nn.Conv2d(in_channels=out_channels * 8, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        x = self.vit(x)
        x = rearrange(x, 'b (x y) c -> b c x y', x=self.vit_img_dim, y=self.vit_img_dim)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x, x1, x2, x3

        pass
