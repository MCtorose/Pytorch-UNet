""" Full assembly of the parts to form the complete network """
import torch
import logging

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# from .unet_parts import *
from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels=n_channels, out_channels=64))

        # 下采样
        self.down1 = (Down(in_channels=64, out_channels=128))
        self.down2 = (Down(in_channels=128, out_channels=256))
        self.down3 = (Down(in_channels=256, out_channels=512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(in_channels=512, out_channels=1024 // factor))

        # 上采样
        self.up1 = (Up(in_channels=1024, out_channels=512 // factor, bilinear=bilinear))
        self.up2 = (Up(in_channels=512, out_channels=256 // factor, bilinear=bilinear))
        self.up3 = (Up(in_channels=256, out_channels=128 // factor, bilinear=bilinear))
        self.up4 = (Up(in_channels=128, out_channels=64, bilinear=bilinear))

        # 输出层
        self.outc = (OutConv(in_channels=64, out_channels=n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        print(f"x1.shape为{x1.shape}")
        # x1 = ECA(channel=x1.shape[1], k_size=3)(x1)
        # x1 = TransformerLayer(input_dim=x1.shape[1], hidden_dim=x1.shape[1], num_heads=4).cuda()(x1)
        x2 = self.down1(x1)
        print(f"x2.shape为{x2.shape}")
        # x2 = ECA(channel=x2.shape[1], k_size=3)(x2)
        # x2 = TransformerLayer(input_dim=x2.shape[1], hidden_dim=x2.shape[1], num_heads=4).cuda()(x2)
        x3 = self.down2(x2)
        print(f"x3.shape为{x3.shape}")
        # x3 = ECA(channel=x3.shape[1], k_size=3)(x3)
        # x3 = TransformerLayer(input_dim=x3.shape[1], hidden_dim=x3.shape[1], num_heads=4).cuda()(x3)
        x4 = self.down3(x3)
        print(f"x4.shape为{x4.shape}")
        # x4 = ECA(channel=x4.shape[1], k_size=3)(x4)
        # x4 = TransformerLayer(input_dim=x4.shape[1], hidden_dim=x4.shape[1], num_heads=4).cuda()(x4)
        x5 = self.down4(x4)
        print(x5.shape)
        print(f"x5.shape为{x5.shape}")

        # 四层跳跃连接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 三层跳跃连接
        # x = self.up2(x4, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)

        logits = self.outc(x)

        return logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=2).to(device)
    input = torch.randn((1, 3, 64, 224)).to(device)
    output = net(input)
    print(output.shape)
