import argparse
import logging
import os
import timeit




from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

import time

time_list = []


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    # 记录开始时间
    start_time = time.time()

    with torch.no_grad():
        output = net(img)
        output = output.cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    time_taken = end_time - start_time
    time_list.append(time_taken)
    # 打印运行时间
    # print(f"Time taken: {time_taken:.4f} seconds")
    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=r'E:\Desktop\Pytorch-UNet\checkpoints\checkpoint_epoch_test_10_16_50.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', default=False, action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', default=False, action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    print(args)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    # img = Image.open('./33.bmp').convert('RGB')
    # mask_test = predict_img(net=net, full_img=img, scale_factor=args.scale, out_threshold=0.5, device=device)
    # out_filename = './3344.png'
    # result = mask_to_image(mask_test, mask_values)
    # result.save(out_filename)
    # logging.info(f'Mask saved to {out_filename}')

    # for i, filename in enumerate(os.listdir(r'E:\train_image\VOC10_11\JPEGImages')):
    #     filename = os.path.join(r'E:\train_image\VOC10_11\JPEGImages', filename)
    #     logging.info(f'Predicting image {filename} ...')
    #     img = Image.open(filename).convert('RGB')
    #
    #     mask = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)
    #
    #     if not args.no_save:
    #         out_filename = filename.replace('JPEGImages', 'predict10_16').replace('.jpg', '.png')
    #         result = mask_to_image(mask, mask_values)
    #         result.save(out_filename)
    #         logging.info(f'Mask saved to {out_filename}')
    #
    #     if args.viz:
    #         logging.info(f'Visualizing results for image {filename}, close to continue...')
    #         plot_img_and_mask(img, mask)

    print(time_list)
