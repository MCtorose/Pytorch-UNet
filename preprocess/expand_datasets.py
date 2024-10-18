import os
import random
from PIL import Image

# 定义路径
image_dir = r'E:\train_image\VOC10_17\JPEGImages'
label_dir = r'E:\train_image\VOC10_17\SegmentationClass'
output_image_dir = r'E:\train_image\VOC10_17\AugmentedImages'
output_label_dir = r'E:\train_image\VOC10_17\AugmentedLabels'

# 创建输出目录（如果不存在）
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 获取所有文件名
image_filenames = os.listdir(image_dir)

# 定义可能的变换
transformations = [
    'flip_left_right',
    'flip_top_bottom'
]


def apply_transformation(image, transformation):
    if transformation == 'flip_left_right':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif transformation == 'flip_top_bottom':
        return image.transpose(Image.FLIP_TOP_BOTTOM)


# 对每个文件进行变换
for filename in image_filenames:
    base_name = os.path.splitext(filename)[0]
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, filename.replace('.jpg', '.png'))
    print(label_path)

    if os.path.exists(label_path):
        # 打开图像和标签
        image = Image.open(image_path)
        label = Image.open(label_path)

        # 随机选择一个变换
        transformation = random.choice(transformations)

        # 应用变换
        transformed_image = apply_transformation(image, transformation)
        transformed_label = apply_transformation(label, transformation).convert('RGB')

        # 保存变换后的图像和标签
        transformed_image.save(os.path.join(output_image_dir, base_name + '_' + transformation + '.jpg'))
        transformed_label.save(os.path.join(output_label_dir, base_name + '_' + transformation + '.png'))

print("数据扩充完成。")
