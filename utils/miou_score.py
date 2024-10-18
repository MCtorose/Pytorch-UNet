import numpy as np
import torch


#
# def calculate_iou(pred, target, n_classes, ignore_background=True):
#     if ignore_background:
#         # 忽略背景类别（假设背景类别为0）
#         pred = pred[target > 0]
#         target = target[target > 0]
#
#     ious = []
#     pred_inds = (pred == 1)
#     target_inds = (target == 1)
#
#     intersection = (pred_inds[target_inds]).long().sum().item()
#     union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
#
#     if union != 0:
#         ious.append(float(intersection) / float(max(union, 1)))
#
#     return np.mean(ious) if ious else 0.0
#
#
# def calculate_mpa(pred, target, n_classes, ignore_background=True):
#     if ignore_background:
#         # 忽略背景类别（假设背景类别为0）
#         pred = pred[target > 0]
#         target = target[target > 0]
#
#     accuracies = []
#     pred_inds = (pred == 1)
#     target_inds = (target == 1)
#
#     true_positive = (pred_inds[target_inds]).long().sum().item()
#     total_target = target_inds.long().sum().item()
#
#     if total_target != 0:
#         accuracies.append(float(true_positive) / float(total_target))
#
#     return np.mean(accuracies) if accuracies else 0.0
#
#
# def calculate_confusion_matrix(pred, target, n_classes, ignore_background=True):
#     if ignore_background:
#         # 忽略背景类别（假设背景类别为0）
#         pred = pred[target > 0]
#         target = target[target > 0]
#
#     confusion_matrix = np.zeros((2, 2), dtype=np.int32)
#     confusion_matrix[0, 0] = ((target == 0) & (pred == 0)).sum()  # 背景的TP
#     confusion_matrix[1, 1] = ((target == 1) & (pred == 1)).sum()  # 焊缝的TP
#     confusion_matrix[0, 1] = ((target == 1) & (pred == 0)).sum()  # 背景的FP
#     confusion_matrix[1, 0] = ((target == 0) & (pred == 1)).sum()  # 焊缝的FN
#
#     return confusion_matrix

# 示例使用
# pred = torch.randint(0, 2, (1, 256, 256))  # 示例预测张量，只有背景和焊缝两个类别
# target = torch.randint(0, 2, (1, 256, 256))  # 示例真实标签张量
#
# iou = calculate_iou(pred, target)
# mpa = calculate_mpa(pred, target)
# confusion_matrix = calculate_confusion_matrix(pred, target)
#
# print("IoU:", iou)
# print("MPA:", mpa)
# print("Confusion Matrix:\n", confusion_matrix)
def calculate_iou(pred, target, n_classes, ignore_background=True):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection

        if union != 0:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.mean(ious)


def calculate_mpa(pred, target, n_classes):
    accuracies = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        true_positive = (pred_inds[target_inds]).long().sum().item()
        total_target = target_inds.long().sum().item()

        if total_target != 0:
            accuracies.append(float(true_positive) / float(total_target))

    return np.mean(accuracies)


def calculate_confusion_matrix(pred, target, n_classes):
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

    pred = pred.view(-1)
    target = target.view(-1)

    for t, p in zip(target, pred):
        confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix

# 示例调用
# pred 和 target 是预测和实际的 tensor，n_classes 是类别数
# mpa = calculate_mpa(pred, target, n_classes)


# from sklearn.metrics import confusion_matrix
#
#
# def compute_mIoU(true, pred, num_classes):
#     confusion = confusion_matrix(true.flatten(), pred.flatten(), labels=np.arange(num_classes))
#     intersection = np.diag(confusion)
#     union = np.sum(confusion, axis=0) + np.sum(confusion, axis=1) - intersection
#     IoU = intersection / union
#     return np.mean(IoU)


# 示例用法
# pred = np.array([[0, 1, 1], [1, 1, 0]])
# target = np.array([[0, 1, 0], [1, 0, 0]])
# n_classes = 2
#
# iou = calculate_iou(pred, target, n_classes)
# iou2 = compute_mIoU(pred, target, n_classes)
# print(f"Mean IoU: {iou}")
# print(f"Mean IoU: {iou2}")

# # 示例用法
# pred = torch.tensor([0, 1, 2, 2, 1])
# target = torch.tensor([0, 1, 1, 2, 2])
# n_classes = 3
#
# print(confusion_matrix(target, pred, labels=np.arange(3)))
#
# print(calculate_mpa(pred, target, n_classes))
# conf_matrix = calculate_confusion_matrix(pred, target, n_classes)
# print(conf_matrix)
