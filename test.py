from PIL import Image
import matplotlib.pyplot as plt

# # 打开图片
# image = Image.open(r'E:\train_image\VOC10_17\SegmentationClass\result_2.png')
#
# # 获取图片的宽度和高度
# width, height = image.size
#
# # 打开一个新的 txt 文件
# with open('pixels3.txt', 'w') as file:
#     # 遍历每个像素
#     for y in range(height):
#         for x in range(width):
#             # 获取像素值
#             pixel = image.getpixel((x, y))
#             # 将像素值写入 txt 文件
#             file.write(f'{pixel}')
#         file.write('\n')

# import cv2
#
# # 读取图片
# image1 = label
# image2 = pre
#
# # 调整图片尺寸
# height, width = image1.shape[:2]
# image2 = cv2.resize(image2, (width, height))
#
# # 图片重叠
# alpha = 0.5  # 图片1的权重
# beta = 0.5   # 图片2的权重
# gamma = 0.0  # 伽马校正值
#
# blended_image = cv2.addWeighted(image1, alpha, image2, beta, gamma)
#
# # 显示结果
# cv2.imshow('Blended Image', blended_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 保存结果
# cv2.imwrite('blended_image2.jpg', blended_image)
