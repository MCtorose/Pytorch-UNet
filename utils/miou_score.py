import numpy as np


def calculate_iou(pred, target, n_classes):
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
