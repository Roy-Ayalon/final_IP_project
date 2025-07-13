import os, glob, csv, math
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm
from unet import UNet                # your model
from dataset import WheatSegDataset  # reuse transforms
from torch.utils.data import DataLoader

# hyperparameters
DEVICE  = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
THRESH  = 0.35          
MIN_BOX = 10             
MODEL_W = "unet_baseline_mps.pth"

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_W, map_location=DEVICE))
model.eval()

# -------------------------------------------------------
# 2. DataLoader for test images
# -------------------------------------------------------
test_imgs = glob.glob("data/test/**/*.jpg", recursive=True)
test_ds   = WheatSegDataset("data/test", masks_dir=None, transforms=None)  # only images
test_dl   = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

# -------------------------------------------------------
# 3. Helper: mask → list of (score, x1,y1,x2,y2)
# -------------------------------------------------------
def masks_to_boxes(prob_mask, thresh=THRESH):
    """
    prob_mask: np.ndarray H×W float32
    Returns list of (score, x1,y1,x2,y2) in pixel coords.
    """
    mask_bin = (prob_mask > thresh).astype(np.uint8)
    cnts, _  = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if max(w,h) < MIN_BOX:
            continue
        x1,y1,x2,y2 = x, y, x+w, y+h
        # confidence = mean probability inside contour
        score = float(prob_mask[y1:y2, x1:x2].mean())
        boxes.append((score,x1,y1,x2,y2))
    return boxes

# -------------------------------------------------------
# 4. Run inference & build submission rows
# -------------------------------------------------------
rows = []
with torch.no_grad():
    for imgs, paths in tqdm(test_dl, desc="Predicting"):
        imgs = imgs.to(DEVICE)
        probs = model(imgs).squeeze(1).cpu().numpy()   # [B,H,W]
        for prob, path in zip(probs, paths):
            boxes = masks_to_boxes(prob)
            # Build PredictionString: "score x1 y1 x2 y2 score x1 y1 ..."
            pred_str = " ".join(
                f"{b[0]:.4f} {b[1]} {b[2]} {b[3]} {b[4]}" for b in boxes
            )
            image_id = os.path.splitext(os.path.basename(path))[0]
            rows.append({"image_id": image_id, "PredictionString": pred_str})

sub_df = pd.DataFrame(rows)
sub_df.to_csv("sub_unet.csv", index=False)
print("Wrote", len(rows), "rows to sub_unet.csv")

# -------------------------------------------------------
# 5. (Optional) Local metric – image precision @ IoU 0.5:0.05:0.75
# -------------------------------------------------------
IOU_THRESHOLDS = np.arange(0.5, 0.76, 0.05)

def iou(box1, box2):
    """IoU of 2 boxes [x1,y1,x2,y2]"""
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    if xB <= xA or yB <= yA:
        return 0.0
    inter = (xB-xA) * (yB-yA)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter)

def image_precision(gt_boxes, pred_boxes, scores):
    """Mean precision over IoU thresholds for one image."""
    if len(pred_boxes)==0: return 0.0
    order = np.argsort(scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in order]
    matched_gt = set()
    precisions = []
    for t in IOU_THRESHOLDS:
        tp = 0
        for pb in pred_boxes:
            matched = False
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt: continue
                if iou(pb, gb) >= t:
                    matched_gt.add(i); matched=True; break
            tp += matched
        fp = len(pred_boxes)-tp
        fn = len(gt_boxes)-len(matched_gt)
        precisions.append(tp / (tp+fp+fn+1e-6))
    return np.mean(precisions)

# Example usage on validation split:
# val_metric = np.mean([image_precision(gt[i], preds[i], confs[i]) for i in range(N)])