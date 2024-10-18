""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class FReLU(nn.Module):
    """FReLU activation https://arxiv.org/abs/2007.11824."""

    def __init__(self, c1, k=3):  # ch_in, kernel
        """Initializes FReLU activation with channel `c1` and kernel size `k`."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        """
        Applies FReLU activation with max operation between input and BN-convolved input.

        https://arxiv.org/abs/2007.11824
        """
        return torch.max(x, self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Up-scaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1).cuda()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class TransformerLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(TransformerLayer, self).__init__()
        # AssertionError: embed_dim must be divisible by num_heads
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=input_dim)
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=input_dim)

    def forward(self, x):
        batch_size, feature_size, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
        # 将输入张量 x 重塑为 (batch_size, W*H, feature_size)，以便通过注意力机制处理。
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(3))

        residual = x
        # 应用层归一化
        x = self.layer_norm(x)
        # 自注意力机制
        x, _ = self.attention(x, x, x)
        # 残差连接
        x = F.relu(x + residual)

        residual = x
        # 再次应用层归一化
        x = self.layer_norm(x)
        # 前馈神经网络
        x = self.feed_forward(x)
        # 再次添加残差连接
        x = F.relu(x + residual)

        # 输出重塑回原始形状 (batch_size, W, H, feature_size)
        # 首先通过 view 方法重塑张量的形状以包含空间维度，
        # 然后通过 permute 方法重新排列这些维度，以满足特定的数据处理或网络层要求。
        # 这在处理图像数据或任何需要特定维度顺序的多维数据时非常有用。
        x = x.view(x.size(0), W, H, x.size(2))
        x = x.permute(0, 3, 1, 2)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x0 = input
        x = F.relu(self.bn1(self.conv1(x0)))
        x = self.bn2(self.conv2(x))
        x = x + self.conv3(x0)
        return F.relu(x, inplace=True)  # Optional: Apply ReLU to the final output


class ResNetDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ResNetBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SimAM(nn.Module):
    def __init__(self, lamda=1e-5):
        super().__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入张量的形状信息
        b, c, h, w = x.shape
        # 计算像素点数量
        n = h * w - 1
        # 计算输入张量在通道维度上的均值
        mean = torch.mean(x, dim=[-2, -1], keepdim=True)
        # 计算输入张量在通道维度上的方差
        var = torch.sum(torch.pow((x - mean), 2), dim=[-2, -1], keepdim=True) / n
        # 计算特征图的激活值
        e_t = torch.pow((x - mean), 2) / (4 * (var + self.lamda)) + 0.5
        # 使用 Sigmoid 函数进行归一化
        out = self.sigmoid(e_t) * x
        return out


# 测试模块
if __name__ == "__main__":
    # 创建 SimAM 实例并进行前向传播测试
    layer = SimAM(lamda=1e-5)
    x = torch.randn((2, 3, 224, 224))
    output = layer(x)
    # print("Output shape:", output.shape)

# if __name__ == '__main__':
#     input = torch.ones(size=(1, 64, 64, 224))
#     model = ResNetDown(in_channels=64, out_channels=128)  # torch.Size([1, 128, 32, 112])
#     output = model(input)
#     print(output)
#     print(output.shape)

# output = TransformerLayer(input_dim=input.shape[1], hidden_dim=input.shape[1], num_heads=3)(input)

# print(output)
# print(output.shape)
