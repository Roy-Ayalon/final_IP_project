
# %% Cell 1: Imports & helpers
import time
import math
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice
from dataset import get_dataloaders   # your custom loader (returns masks)
from unet import UNet

# ─── Competition metric helpers ────────────────────────────────────────────
IOU_THRESHOLDS = np.arange(0.50, 0.76, 0.05)

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

def image_precision(gt_boxes, pred_boxes, scores):
    """mean AP for ONE image as defined by Kaggle."""
    order = np.argsort(scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in order]
    scores = [scores[i] for i in order]
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

def mask_to_boxes(mask_bin, min_size=10):
    """binary uint8 mask → list of [x1,y1,x2,y2]"""
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if max(w,h) < min_size: continue
        boxes.append([x,y,x+w,y+h])
    return boxes

# ─── Loss: BCE + Dice ──────────────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = Dice(ignore_index=None)

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        dice = 1 - self.dice(preds, targets.int())
        return bce + dice

# %% Cell 2: Main training script
if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0)
    print(f"Train imgs: {len(train_loader.dataset)} | Val imgs: {len(val_loader.dataset)}")

    model = UNet().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    NUM_EPOCHS = 20
    history = {"train_loss": [], "val_map": []}

    for epoch in range(1, NUM_EPOCHS+1):
        t0 = time.time()
        # ── Train ──────────────────────────────────
        model.train()
        running = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss  = criterion(preds, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # ── Validate ───────────────────────────────
        model.eval()
        all_prec = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs).cpu().numpy()
                masks_np = masks.cpu().numpy()

                for pr, gt in zip(preds, masks_np):
                    pr_bin = (pr[0] > 0.5).astype(np.uint8)
                    gt_bin = gt[0].astype(np.uint8)

                    pred_boxes = mask_to_boxes(pr_bin)
                    scores = [pr[0][b[1]:b[3], b[0]:b[2]].mean() for b in pred_boxes]

                    gt_boxes = mask_to_boxes(gt_bin)
                    if len(gt_boxes)==0 and len(pred_boxes)==0:
                        all_prec.append(1.0)  # perfect empty
                        continue
                    all_prec.append(image_precision(gt_boxes, pred_boxes, scores))

        val_map = float(np.mean(all_prec))
        history["val_map"].append(val_map)
        dt = time.time()-t0

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"loss={train_loss:.4f} | mAP={val_map:.5f} | {dt:.1f}s")

    torch.save(model.state_dict(), "unet_baseline_mps.pth")
    print("Weights saved → unet_baseline_mps.pth")