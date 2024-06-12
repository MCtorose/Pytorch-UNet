import torch
import torch.nn as nn

# 定义最大池化层，步幅为1
max_pool = nn.MaxPool2d(kernel_size=2, stride=1)

# 输入张量：批量大小为1，通道数为1，高度为14，宽度为125
input_tensor = torch.randn(1, 1, 32, 125)

# 进行五次最大池化操作
for _ in range(5):
    input_tensor = max_pool(input_tensor)

print(input_tensor.shape)