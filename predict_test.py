import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from utils.data_loading import BasicDataset, CarvanaDataset
from unet import UNet
from utils.miou_score import calculate_iou, calculate_mpa, calculate_confusion_matrix
from utils.utils import plot_img_and_mask


def predict_img(net, full_img, true_mask, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        output = output.cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

        mask_pred = mask[0].square()
        true_mask = torch.from_numpy(np.array(true_mask))
        # 计算miou
        iou = calculate_iou(true_mask, mask_pred, net.n_classes)
        # 计算mpa
        mpa = calculate_mpa(true_mask, mask_pred, net.n_classes)
        # 计算混淆矩阵
        cm = calculate_confusion_matrix(true_mask, mask_pred, net.n_classes)

    return mask[0].long().squeeze().numpy(), iou, mpa, cm


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

    # # 1. 创建数据集
    # try:
    #     dataset = CarvanaDataset(images_dir=dir_img, mask_dir=dir_mask, scale=img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)
    #
    # # 2. 创建数据加载器
    # loader_args = dict(batch_size=8, num_workers=0, pin_memory=True)
    # test_loader = DataLoader(dataset, shuffle=True, **loader_args)
    # with torch.no_grad():
    #     net.eval()
    #     for batch in test_loader:
    #         images, true_masks = batch['image'], batch['mask']
    #         logging.info(f'Predicting on [INFO] images.shape: {images.shape}')
    #         logging.info(f'Predicting on [INFO] true_masks.shape: {true_masks.shape}')
    #         # 确保图像和掩膜的维度正确
    #         assert images.shape[1] == net.n_channels, \
    #             f'Network has been defined with {net.n_channels} input channels, ' \
    #             f'but loaded images have {images.shape[1]} channels. Please check that ' \
    #             'the images are loaded correctly.'
    #         # 将图像和掩膜移动到设备上
    #         images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
    #         true_masks = true_masks.to(device=device, dtype=torch.long)
    #         masks_pred = net(images)
    #         pred = torch.argmax(F.softmax(masks_pred, dim=1), dim=1) if net.n_classes > 1 else (F.sigmoid(masks_pred) > 0.5).long()
    #         miou = calculate_iou(pred=pred, target=true_masks, n_classes=net.n_classes)
    #         mpa = calculate_mpa(pred=pred, target=true_masks, n_classes=net.n_classes)
    #         confusion_matrix = calculate_confusion_matrix(pred=pred, target=true_masks, n_classes=net.n_classes)
    #         logging.info(f'Predicting on [INFO] miou: {miou}')
    #         logging.info(f'Predicting on [INFO] mpa: {mpa}')
    #         logging.info(f'Predicting on [INFO] confusion_matrix: {confusion_matrix}')
    #
    # img = Image.open('./33.bmp').convert('RGB')
    # mask_test = predict_img(net=net, full_img=img, scale_factor=args.scale, out_threshold=0.5, device=device)
    # out_filename = './3344.png'
    # result = mask_to_image(mask_test, mask_values)
    # result.save(out_filename)
    # logging.info(f'Mask saved to {out_filename}')

    for i, filename in enumerate(os.listdir(r'E:\train_image\VOC10_17\VOC_17_test\JPEGImages')):
        filename = os.path.join(r'E:\train_image\VOC10_17\VOC_17_test\JPEGImages', filename)
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename).convert('RGB')
        true_mask = Image.open(filename.replace('JPEGImages', 'SegmentationClass').replace('.jpg', '.png')).convert('P')

        result = predict_img(net=net,
                             full_img=img,
                             true_mask=true_mask,
                             scale_factor=args.scale,
                             out_threshold=args.mask_threshold,
                             device=device)

        mask, iou, mpa, cm = result
        A = cm[0][0]
        B = cm[0][1]
        C = cm[1][0]
        D = cm[1][1]

        print(cm, A)
        if not args.no_save:
            out_filename = filename.replace('JPEGImages', 'predict10_17').replace('.jpg', '.png')
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
