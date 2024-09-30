# import torch
# import torch.nn as nn
#
# # 定义最大池化层，步幅为1
# max_pool = nn.MaxPool2d(kernel_size=2, stride=1)
#
# # 输入张量：批量大小为1，通道数为1，高度为14，宽度为125
# input_tensor = torch.randn(1, 1, 32, 125)
#
# # 进行五次最大池化操作
# for _ in range(5):
#     input_tensor = max_pool(input_tensor)
#
# print(input_tensor.shape)
#
#

import base64
import cv2
import numpy as np

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 读取labelme生成的JSON文件
with open(r'E:\Desktop\Matlab_DeepL\split_png\json\result_1.json') as f:
    data = json.load(f)

# 提取点信息
# print(data)

# # 获取Base64编码的图像数据
# encoded_image_data = data['imageData']
#
# # 解码Base64数据为二进制图像数据
# image_data = base64.b64decode(encoded_image_data)
#
# # 将二进制图像数据转为NumPy数组
# image_array = np.frombuffer(image_data, dtype=np.uint8)
#
# # 使用OpenCV将NumPy数组解码为图像
# image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#
# # 显示图像
# cv2.imshow('Decoded Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


shape = data['shapes']  # 假设只有一个形状
points = shape['points']
# print(points)
#
# for point in points:
#     print(point)
#
#
#
# # 绘制多边形
# fig, ax = plt.subplots()
# polygon = patches.Polygon(points, closed=True, edgecolor='r', facecolor='none')
# ax.add_patch(polygon)
# ax.set_xlim([0, 224])
# ax.set_ylim([0, 64])
# # plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 给定点列表
points = np.array([
    [144, 21], [145, 8], [145, 20], [145, 22], [145, 23], [145, 24], [145, 25], [145, 26],
    [145, 27], [145, 28], [145, 43], [145, 44], [145, 45], [145, 46], [145, 47], [145, 48],
    [145, 49], [145, 50], [145, 51], [145, 52], [145, 53], [145, 54], [145, 55], [145, 56],
    [145, 57], [145, 58], [145, 59], [145, 60], [145, 61], [145, 62], [145, 63], [145, 64],
    [146, 1], [146, 2], [146, 3], [146, 4], [146, 5], [146, 6], [146, 7], [146, 9], [146, 10],
    [146, 11], [146, 12], [146, 13], [146, 14], [146, 15], [146, 16], [146, 17], [146, 18],
    [146, 19], [146, 29], [146, 30], [146, 31], [146, 32], [146, 33], [146, 34], [146, 35],
    [146, 36], [146, 37], [146, 38], [146, 39], [146, 40], [146, 41], [146, 42], [146, 64],
    [147, 1], [147, 64], [148, 1], [148, 64], [149, 1], [149, 64], [150, 1], [150, 64], [151, 1],
    [151, 12], [151, 13], [151, 14], [151, 15], [151, 16], [151, 17], [151, 18], [151, 19], [151, 20],
    [151, 21], [151, 22], [151, 23], [151, 32], [151, 33], [151, 34], [151, 35], [151, 36], [151, 37],
    [151, 64], [152, 1], [152, 3], [152, 4], [152, 5], [152, 6], [152, 7], [152, 8], [152, 11], [152, 24],
    [152, 31], [152, 38], [152, 39], [152, 40], [152, 41], [152, 42], [152, 46], [152, 47], [152, 48],
    [152, 49], [152, 50], [152, 51], [152, 52], [152, 53], [152, 54], [152, 55], [152, 56], [152, 57],
    [152, 58], [152, 59], [152, 60], [152, 61], [152, 62], [152, 63], [152, 64], [153, 1], [153, 2],
    [153, 9], [153, 10], [153, 25], [153, 26], [153, 30], [153, 43], [153, 45], [153, 64], [154, 27],
    [154, 28], [154, 29], [154, 30]
])

# 计算质心
centroid = np.mean(points, axis=0)

# 计算每个点到质心的距离
distances = np.linalg.norm(points - centroid, axis=1)

# 计算距离的平均值和标准差
mean_distance = np.mean(distances)
std_distance = np.std(distances)

# 设置过滤阈值，保留在平均距离±2倍标准差内的点
threshold = 2 * std_distance
filtered_points = points[(distances >= (mean_distance - threshold)) & (distances <= (mean_distance + threshold))]

# 计算每个点相对于质心的角度
angles = np.arctan2(filtered_points[:, 1] - centroid[1], filtered_points[:, 0] - centroid[0])

# 按角度对点进行排序
sorted_points = filtered_points[np.argsort(angles)]

# 绘制排序前后的点
plt.figure(figsize=(12, 6))

# 左图：未排序点
plt.subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], color='blue')
plt.plot(np.append(points[:, 0], points[0, 0]), np.append(points[:, 1], points[0, 1]), linestyle='-', color='blue')
plt.title('Before Filtering and Sorting')
plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates

# 右图：已排序点
plt.subplot(1, 2, 2)
plt.scatter(sorted_points[:, 0], sorted_points[:, 1], color='red')
plt.plot(np.append(sorted_points[:, 0], sorted_points[0, 0]), np.append(sorted_points[:, 1], sorted_points[0, 1]), linestyle='-', color='red')
plt.title('After Filtering and Sorting')
plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates

plt.show()

print("Filtered and Sorted Points:\n", sorted_points)

