import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.miou_score import calculate_iou
import pandas as pd

data = {
    'epoch': [],
    'step': [],
    'train_loss': [],
    'dice_score': [],
    'miou': [],
}

# 文件路径声明
dir_img = r'E:\train_image\VOC2\JPEGImages'
dir_mask = r'E:\train_image\VOC2\SegmentationClass'
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 20,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. 创建数据集
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset=dataset, lengths=[n_train, n_val], generator=torch.Generator().manual_seed(0))

    logging.info(f"[INFO] Number of samples: {len(dataset)}")
    logging.info(f"[INFO] Number of training samples: {len(train_set)}")
    logging.info(f"[INFO] Number of validation samples: {len(val_set)}")

    # 3. 创建loader
    # 当设置为 True 时，此选项告诉数据加载器使用固定内存，这可以提高数据传输到 GPU 的性能。
    # 固定内存是锁定在 RAM 中的内存，不能被分页到磁盘，这允许更快地传输到 GPU 内存。
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    logging.info(f"[INFO] Using loader_args: {loader_args}")

    train_loader = DataLoader(dataset=train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(dataset=val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must', name="ECA_unet")
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''
        Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision 混合精度: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: 最大化dice的值
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    # 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        miou_scores = []

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                logging.info(f"[INFO] images.shape: {images.shape}")
                logging.info(f"[INFO] true_masks.shape: {true_masks.shape}")
                images = images.repeat(1, 3, 1, 1)

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                # 放入GPU设备
                # 指定张量在内存中的布局格式。
                # torch.channels_last 是一种内存布局，它将通道维度放在最后，即 (batch_size, height, width, channels)。
                # 这种布局通常用于与 GPU 兼容的内存格式，可以提高某些操作的性能。
                # 与之相对的是 torch.contiguous_format，它将通道维度放在第二个位置，即 (batch_size, channels, height, width)
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # 计算损失函数
                # 启用自动混合精度  mps  "Metal Performance Shaders" 的缩写，
                # Apple 提供的一种框架，用于在 macOS 和 iOS 设备上高效地执行图形和计算任务,可以在 GPU 上执行，从而加速图形渲染和计算密集型任务。
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    logging.info(f"[INFO] model.n_classes: {model.n_classes}")
                    # 如果模型只有一个类别，那么处理的是二分类问题，通常使用二元交叉熵损失（Binary Cross Entropy, BCE）和 Dice 损失的组合。
                    # F.sigmoid 用于将预测掩码的值映射到 [0, 1] 区间
                    # F.softmax 用于将预测掩码的值转换为概率分布，
                    # F.one_hot 用于将真实掩码转换为 one-hot 编码
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                # 用于清零模型参数的梯度。在 PyTorch 中，每次反向传播之前都需要清零之前的梯度，以避免梯度累积
                # set_to_none 参数设置为 True 时，会将梯度设置为 None，这通常比将梯度设置为零更快，因为它避免了创建新的张量。
                optimizer.zero_grad(set_to_none=True)
                # 使用梯度缩放器（grad_scaler）来缩放损失值，然后进行反向传播计算梯度。
                # 梯度缩放器是自动混合精度训练中的一个组件，用于缩放损失值以避免在反向传播过程中出现下溢（underflow）。
                # 缩放后的损失值用于计算模型参数的梯度。
                grad_scaler.scale(loss).backward()
                # 用于取消梯度缩放器对优化器的缩放。
                # 在调用 optimizer.step() 之前，需要取消梯度缩放，以便优化器可以正确地更新模型参数。
                grad_scaler.unscale_(optimizer)
                # 用于梯度裁剪（gradient clipping），这是一种防止梯度爆炸的技术。
                # 它限制了模型参数梯度的范数，如果梯度的范数超过了设定的阈值 gradient_clipping，则将其缩放到不超过该阈值。
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=gradient_clipping)
                # 更新模型参数
                # 梯度缩放器在这里的作用是确保在更新参数之前，梯度已经被正确地缩放和取消缩放。
                grad_scaler.step(optimizer)
                # 用于更新梯度缩放器的缩放因子。
                # 在每次参数更新之后，梯度缩放器会根据前向传播和反向传播的结果来调整缩放因子，以便在下一次迭代中使用。
                grad_scaler.update()
                # 更新进度条
                pbar.update(n=images.shape[0])
                # 于增加全局步数计数器。global_step 是一个用于跟踪训练过程中总步数的变量。
                global_step += 1
                # 单步求得的训练损失
                epoch_loss += loss.item()
                logging.info(f'[INFO] train loss: {loss.item()}, step: {global_step}, epoch: {epoch}')
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                # tqdm 进度条对象的一个方法，用于在进度条的后缀部分显示额外的信息。
                # 这里通过关键字参数的形式传入了一个字典，字典的键是 'loss (batch)'，表示要显示的标签，值是 loss.item()，表示当前批次的损失值。
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # 单步求得的训练损失
                # mIoU计算
                pred = torch.argmax(F.softmax(masks_pred, dim=1), dim=1) if model.n_classes > 1 else (F.sigmoid(masks_pred) > 0.5).long()
                miou = calculate_iou(pred, true_masks, model.n_classes)
                miou_scores.append(miou)

                data['epoch'].append(epoch)
                data['step'].append(global_step)
                data['train_loss'].append(loss.item())
                data['miou'].append(miou)

                # 将训练数据存入excel文件

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in model.named_parameters():
                        # tag = tag.replace('/', '.')
                        # if not (torch.isinf(value) | torch.isnan(value)).any():
                        #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        # if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # 验证集
                        val_score = evaluate(net=model, dataloader=val_loader, device=device, amp=amp)
                        # 调用了学习率调度器（scheduler）的 step 方法，并传入验证分数。学习率调度器可能根据验证分数调整学习率，以优化模型性能。
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        data['dice_score'].append(val_score.item())
                        # try:
                        #     experiment.log({
                        #         'learning rate': optimizer.param_groups[0]['lr'],
                        #         'validation Dice': val_score,
                        #         'images': wandb.Image(images[0].cpu()),
                        #         'masks': {
                        #             'true': wandb.Image(true_masks[0].float().cpu()),
                        #             'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                        #         },
                        #         'step': global_step,
                        #         'epoch': epoch,
                        #         **histograms
                        #     })
                        # except:
                        #     pass

        avg_miou = np.mean(miou_scores)
        logging.info(f'[INFO] Epoch {epoch} averaged mIoU: {avg_miou}')
        # 保存pth文件
        if save_checkpoint and epoch % 10 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            # 码获取模型的状态字典（state dictionary）。状态字典是 PyTorch 模型的一种表示形式，它包含了模型中所有可学习参数（权重和偏置）的当前状态。
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    # 'epoch': [],
    # 'step': [],
    # 'train_loss': [],
    # 'dice_score': [],
    # 'miou': [],
    # 保存训练数据到excel文件
    print(data)
    # 计算最长的列表长度
    max_length = max(len(data[key]) for key in data)

    # 填充缺失的数据
    for key in ['dice_score', 'miou']:
        if len(data[key]) < max_length:
            data[key] += [np.nan] * (max_length - len(data[key]))

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 保存到Excel文件
    df.to_excel('./result/UNET_100epoch_8batch_size_output.xlsx', index=False)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    # 混合精度训练是一种在深度学习中使用的技术，它结合了半精度（例如，16位浮点数，即FP16）和单精度（32位浮点数，即FP32）来加速训练过程并减少内存使用。
    # 通过在命令行中使用 --amp 选项，用户可以轻松地控制是否在训练过程中启用混合精度。
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    # 此参数的目的在于指定在执行某些操作（例如图像缩放或上采样）时是否使用双线性插值。双线性插值是一种用于增加图像或其他数据分辨率的像素值插值方法。
    # 它在图像处理和计算机视觉任务中常用。
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # if args.load:
    #     state_dict = torch.load(args.load, map_location=device)
    #     del state_dict['mask_values']
    #     model.load_state_dict(state_dict)
    #     logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    print(device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
