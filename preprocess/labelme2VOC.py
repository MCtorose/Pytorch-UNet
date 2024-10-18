#!/usr/bin/env python
# 在Python 2.x中允许你使用Python 3.x的print函数语法
from __future__ import print_function
# argparse 用于处理命令行参数。
# glob 用于查找符合特定规则的文件路径名。
# os 和 os.path 用于文件和目录操作。
# sys 用于与Python解释器交互。
# imgviz 和 numpy 用于图像处理和数据操作。
# labelme 相关的模块用于处理标注文件。
import argparse
import glob
import os
import os.path as osp
import sys
import imgviz
import numpy as np

from labelme import utils, LabelFile


def main(args):
    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    # 创建文件夹
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClassnpy"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    if not args.noviz:
        os.makedirs(
            osp.join(args.output_dir, "SegmentationClassVisualization")
        )
    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels, encoding='utf-8').readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)
    out_viz_file = ''
    # 遍历所有的json文件
    for filename in glob.glob(pathname=osp.join(args.input_dir, "*.json")):
        print("Generating dataset from:", filename)
        # 标签文件对象
        label_file = LabelFile(filename=filename)

        # 获取文件名称，不包括后缀
        base = osp.splitext(osp.basename(filename))[0]
        # 拼接四个输出路径
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        out_lbl_file = osp.join(
            args.output_dir, "SegmentationClassnpy", base + ".npy"
        )
        out_png_file = osp.join(
            args.output_dir, "SegmentationClass", base + ".png"
        )
        if not args.noviz:
            out_viz_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )
        # 以二进制形式写图片 JPEGImages原图
        with open(out_img_file, "wb") as f:
            f.write(label_file.imageData)
        # 将图片的包含的是图像数据的 base64 编码。为了将其转换为图像数组
        img = utils.img_data_to_arr(img_data=label_file.imageData)

        # 将标注文件中的形状信息转换为两个图像：cls（类别图像）和 ins（实例图像）。
        lbl, _ = utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        # 保存SegmentationClass图片
        utils.lblsave(filename=out_png_file, lbl=lbl)
        # 保存SegmentationClassnpy文件
        np.save(file=out_lbl_file, arr=lbl)
        # 保存可视化文件
        if not args.noviz:
            viz = imgviz.label2rgb(
                # 将标签图像转换为 RGB 图像，并可选地叠加在原始图像上
                label=lbl, image=img, font_size=15, label_names=list(class_names), loc="rb"
            )
            # utils.numpy_to_pillow(arr).save(filename)
            imgviz.io.imsave(filename=out_viz_file, arr=viz)


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_dir", default="E:/600_jpg/imgs/label/labelme/jsons", type=str, help="input annotated directory")
    # parser.add_argument("--output_dir", default="E:/00jpg", type=str, help="output dataset directory")
    # parser.add_argument("--labels", default="E:/600_jpg/imgs/label/labelme/labels.txt", type=str, help="labels file")

    # json文件夹的路径
    parser.add_argument("--input_dir", default=r"E:\Desktop\Matlab_DeepL\split_png\json", type=str, help="input annotated directory")
    # 输出VOC数据集保存路径
    parser.add_argument("--output_dir", default=r"E:\train_image\VOC10_17", type=str, help="output dataset directory")
    parser.add_argument("--labels", default="./labels.txt", type=str, help="labels file")
    parser.add_argument("--noviz", help="no visualization", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
