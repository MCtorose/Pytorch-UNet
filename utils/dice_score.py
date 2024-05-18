import torch
from torch import Tensor


# 计算 Dice 系数的平均值。Dice 系数是一种集合相似度度量函数，通常用于计算两个样本的相似度，取值范围在 0 到 1 之间。
# 在这个函数中，input 是模型预测的分割结果，target 是真实的分割标签。
# reduce_batch_first 参数决定是否在计算 Dice 系数时考虑批次维度。
# epsilon 是一个小的常数，用于防止除以零的错误。
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

# 这个函数计算多类别分割任务中的 Dice 系数的平均值。它通过将输入和目标张量展平来处理多类别情况。
def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

# 这个函数计算 Dice 损失，它是 Dice 系数的补数（1 - Dice 系数）。
# Dice 损失是分割任务中常用的损失函数，用于训练模型。
# multiclass 参数决定是否在多类别情况下计算 Dice 损失。
def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
