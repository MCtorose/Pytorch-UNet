# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
# from unet import UNet
# from unet.unet_parts import *
# from PIL import Image
# import torchvision.transforms as transforms
#
# # 定义图像预处理步骤
# preprocess = transforms.Compose([
#     transforms.ToTensor(),  # 转换为张量
# ])
# torch.manual_seed(10)
# # 加载图像
# image_path = r'E:\Desktop\Pytorch-UNet\0001.jpg'  # 替换为你的图像路径
# image = Image.open(image_path)
#
# # 应用预处理
# input_tensor = preprocess(image)
# # 创建一个批处理维度，通常模型期望输入是4维张量 (batch_size, channels, height, width)
# input_batch = input_tensor.unsqueeze(0)
#
#
# # 定义一个简单的卷积模型
# class SimpleConvNet(nn.Module):
#     def __init__(self):
#         super(SimpleConvNet, self).__init__()
#         self.inc = (DoubleConv(in_channels=1, out_channels=64))
#         # 下采样
#         self.down1 = (Down(in_channels=64, out_channels=128))
#
#     def forward(self, x):
#         x = self.inc(x)
#         x = self.down1(x)
#         return x
#
#
# # 创建模型实例
# model = SimpleConvNet()
# model.eval()  # 设置模型为评估模式
#
# # 执行前向传播
# with torch.no_grad():  # 不需要计算梯度
#     feature_maps = model(input_batch)
#
# # 将特征图从GPU转移到CPU，并转换为numpy数组
# feature_maps = feature_maps.cpu().numpy()
#
# # 假设我们只绘制第一个特征图
# first_feature_map = feature_maps[0, 0, :, :]
#
# # 将特征图缩放到0-255范围，并转换为8位无符号整数
# first_feature_map = (first_feature_map - first_feature_map.min()) / (first_feature_map.max() - first_feature_map.min()) * 255
# first_feature_map = first_feature_map.astype(np.uint8)
#
# # 创建一个子图，显示原图和特征图
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
# # 在第一个子图中显示原图
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title('Original Image')
# axes[0].axis('off')
#
# # 在第二个子图中显示特征图
# axes[1].imshow(first_feature_map, cmap='gray')
# axes[1].set_title('Feature Map after Convolution')
# axes[1].axis('off')
#
# plt.tight_layout()
# plt.show()


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# 定义一个简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # 输入通道1，输出通道6，卷积核大小3x3
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，大小2x2
        self.conv2 = nn.Conv2d(6, 16, 3)  # 输入通道6，输出通道16，卷积核大小3x3
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 创建网络实例
net = SimpleCNN()

# 用于保存特征图的全局变量
feature_maps = []


# 定义 hook 函数
def forward_hook(module, inputs, outputs):
    feature_maps.append(outputs.cpu().detach().numpy())


# 在 conv1 层上注册 hook
hook_handle = net.conv1.register_forward_hook(hook=forward_hook)

# 生成一个随机输入
input_tensor = torch.randn(1, 1, 32, 32)  # 假设输入是 32x32 的单通道图像

# 运行网络
output = net(input_tensor)

# 移除 hook
hook_handle.remove()


# 绘制特征图
def plot_feature_maps(feature_maps):
    num_feature_maps = feature_maps.shape[1]
    fig, axes = plt.subplots(1, num_feature_maps, figsize=(num_feature_maps * 2, 2))
    if num_feature_maps == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(feature_maps[0, i])
        ax.axis('off')
    plt.show()


# 绘制 conv1 层的特征图
plot_feature_maps(feature_maps[0])
