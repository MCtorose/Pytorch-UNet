import cv2
import numpy as np

# 读取彩色图像
img = cv2.imread(filename='result5.png')

# 假设我们要查看红色通道（索引2）
red_channel = img[:, :, 2]

print(img.shape)
# 遍历红色通道的每个像素
height, width = red_channel.shape
x_list = []
xy_list = []


for y in range(height):
    for x in range(width):
        # 获取像素值并打印
        pixel_value = red_channel[y, x]
        # print(f"Pixel at ({x}, {y}): {pixel_value}")
        if pixel_value == 255:
            x_list.append(x)
    xy_list.append((np.mean(x_list), y))

print(xy_list)
for x, y in xy_list:
    x = int(x)
    y = int(y)
    # 使用cv2.circle函数绘制点
    cv2.circle(img, (x, y), 1, (0, 0, 120), -1)  # 5是点的大小，(0, 0, 255)是红色，-1表示填充颜色

# 显示图像
cv2.imshow('Image with Points', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图像
cv2.imwrite('result3_with_points.png', img)
