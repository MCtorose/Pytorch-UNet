import cv2
import numpy as np

img = cv2.imread(filename=r'E:\train_image\111.jpg')
red_channel = img[:, :, 2]
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
for x, y in xy_list:
    x = int(x)
    y = int(y)
    print(f"({x}, {y})")

    if y % 5 == 0:
        cv2.circle(img, (x, y), 1, (0, 0, 120), -1)  # 5是点的大小，(0, 0, 255)是红色，-1表示填充颜色
cv2.imshow('Image with Points', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result3_with_points.png', img)
