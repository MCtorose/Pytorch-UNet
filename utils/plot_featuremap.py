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
