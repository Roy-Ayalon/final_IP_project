
# %% Cell 1: Imports & helpers
import time
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.segmentation import DiceScore
from unet import UNet

IOU_THRESHOLDS = np.arange(0.50, 0.76, 0.05)

def to_uint8(mask, thr=0.5):
    """
    Ensure a 0/1 uint8 mask – converts floats and bools correctly.
    """
    # squeeze batch dim if present
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    if mask.dtype == np.uint8:
        return (mask > 0).astype(np.uint8)  
    if np.issubdtype(mask.dtype, np.floating):
        return (mask >= thr).astype(np.uint8)
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)

    raise ValueError(f"Unsupported mask dtype: {mask.dtype}")

def mask_to_boxes(mask_bin, min_size=10):
    """binary uint8 mask → list of [x1,y1,x2,y2]"""
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if max(w, h) < min_size:
            continue
        boxes.append([x, y, x + w, y + h])
    return boxes

def iou(box1, box2):
    """IoU of 2 boxes [x1,y1,x2,y2]"""
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    if xB <= xA or yB <= yA:
        return 0.0
    inter = (xB-xA) * (yB-yA)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter)

def image_precision(gt_mask, pred_mask, show=False):
    """mean AP for ONE image as defined by Kaggle."""
    gt_mask = to_uint8(gt_mask)
    pred_mask = to_uint8(pred_mask)
    pred_boxes = mask_to_boxes(pred_mask)
    gt_boxes = mask_to_boxes(gt_mask)
    if show:
        pred_mask_example = pred_mask[0]
        gt_mask_example = gt_mask[0]
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(gt_mask_example, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask_example, cmap='gray')
        plt.title("Predicted Mask")
        plt.show()
    image_precisions = []
    for t in IOU_THRESHOLDS:
        matched_gt = set()
        tp = 0
        for pb in pred_boxes:
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt: continue
                if iou(pb, gb) >= t:
                    matched_gt.add(j)
                    tp += 1
                    break
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)
        image_precisions.append(tp / (tp + fp + fn + 1e-6))
    return np.mean(image_precisions)

def mask_to_centroid_boxes(mask_bin: np.ndarray, min_size: int = 10, pad: int | float = 0, clip: bool = True) -> list[list[int]]:
    """
    Convert a binary (uint8/0-1) mask to bounding boxes whose edges are
    2 × the maximum centroid-to-contour distance along x and y.
    """
    assert mask_bin.ndim == 2, "mask must be H×W, squeeze batch first"
    H, W = mask_bin.shape[:2]
    mask_bin = (mask_bin > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    boxes = []
    for cnt in contours:
        if cnt.shape[0] < 4:           # too small to be meaningful
            continue

        # compute blob moments → centroid (cx, cy) in float px
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

        # longest distance from centroid to ANY contour point along x / y
        dx = np.max(np.abs(cnt[:, 0, 0] - cx))
        dy = np.max(np.abs(cnt[:, 0, 1] - cy))

        if max(dx, dy) < min_size / 2:
            continue

        if isinstance(pad, int):
            dx += pad
            dy += pad
        elif isinstance(pad, float):
            dx *= (1 + pad)
            dy *= (1 + pad)
        else:
            raise TypeError("pad must be int or float")

        # build bbox centred on the centroid
        x1, y1 = cx - dx, cy - dy
        x2, y2 = cx + dx, cy + dy

        if clip:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)

        boxes.append([int(round(x1)),
                      int(round(y1)),
                      int(round(x2)),
                      int(round(y2))])

    return boxes

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceScore(ignore_index=None)

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        dice = 1 - self.dice(preds, targets.int())
        return bce + dice

