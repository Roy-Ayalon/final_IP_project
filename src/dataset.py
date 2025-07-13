import os
from pathlib import Path
import ast
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2
from definitions import *
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ────────────────────── helpers ──────────────────────
def coco_to_cxcywh_norm(boxes, w, h):
    """
    COCO [xmin, ymin, bw, bh] → normalised [cx, cy, bw, bh].
    """
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    boxes = boxes.clone()
    boxes[:, 0] += boxes[:, 2] / 2.0
    boxes[:, 1] += boxes[:, 3] / 2.0
    boxes[:, [0, 2]] /= w
    boxes[:, [1, 3]] /= h
    return boxes


def _bbox_params():
    return A.BboxParams(format="coco",
                        min_visibility=0.3,
                        label_fields=["labels"])


def _post():
    """Resize + tensor conversion *shared* by every pipeline."""
    return [
        A.Resize(256, 256),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ]


# ────────────────────── single-step augs ──────────────────────
def aug_identity():
    return A.Compose(_post(), bbox_params=_bbox_params())


def aug_hflip():
    return A.Compose([A.HorizontalFlip(p=1.0), *_post()],
                     bbox_params=_bbox_params())


def aug_vflip():
    return A.Compose([A.VerticalFlip(p=1.0), *_post()],
                     bbox_params=_bbox_params())


def aug_crop():
    return A.Compose([
        A.CenterCrop(768, 768, p=1.0),
        *_post()
    ], bbox_params=_bbox_params())


def aug_noise():
    return A.Compose([A.GaussNoise(var_limit=(10, 50), p=1.0), *_post()],
                     bbox_params=_bbox_params())


def aug_blur():
    return A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0), *_post()],
                     bbox_params=_bbox_params())

# ────────────────────── combos (same tweak) ──────────────────────
def build_transforms():
    single = {
        "orig":  aug_identity(),
        "hflip": aug_hflip(),
        "vflip": aug_vflip(),
        "crop":  aug_crop(),
        "noise": aug_noise(),
        "blur":  aug_blur(),
    }

    combos = {
        "hflip+noise": A.Compose([
            A.HorizontalFlip(p=1.0),
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            *_post()
        ], bbox_params=_bbox_params()),

        "crop+blur": A.Compose([
            A.CenterCrop(768, 768, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            *_post()
        ], bbox_params=_bbox_params()),
        "hflip+vflip": A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            *_post()
        ], bbox_params=_bbox_params()),
        "vflip+crop+noise": A.Compose([
            A.VerticalFlip(p=1.0),
            A.CenterCrop(768, 768, p=1.0),
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            *_post()
        ], bbox_params=_bbox_params())
    }

    return {**single, **combos}


# ────────────────────── dataset ──────────────────────
class WheatSegDatasetDETRMultiAug(Dataset):
    """
    Each (image × augmentation) pair → one sample, images returned at 256×256.
    """
    def __init__(self, csv_path, images_dir, transforms_dict=None):
        self.images_dir = Path(images_dir)
        self.tfms = transforms_dict or build_transforms()

        df = pd.read_csv(csv_path)
        df["bbox"] = df["bbox"].apply(ast.literal_eval)
        self.boxes = df.groupby("image_id")["bbox"].apply(list).to_dict()
        self.image_ids = sorted([p.stem for p in self.images_dir.glob("*.jpg")])

        self._index = [(i, t) for i in range(len(self.image_ids))
                             for t in self.tfms.keys()]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        img_idx, tname = self._index[idx]
        image_id = self.image_ids[img_idx]
        img_path = self.images_dir / f"{image_id}.jpg"

        image = Image.open(img_path).convert("RGB")
        bboxes = self.boxes.get(image_id, [])
        labels = [0] * len(bboxes)

        t = self.tfms[tname](image=np.array(image),
                             bboxes=bboxes,
                             labels=labels)

        img_t = t["image"]                                   # [C,256,256]
        boxes_t = torch.as_tensor(t["bboxes"], dtype=torch.float32)

        boxes_t = coco_to_cxcywh_norm(boxes_t, 256, 256)

        target = {
            "boxes": boxes_t,
            "labels": torch.zeros(len(boxes_t), dtype=torch.long),
            "image_id": f"{image_id}|{tname}",
            "orig_size": torch.tensor([256, 256]),
        }
        return img_t, target

    @staticmethod
    def collate_fn(batch):
        imgs, targets = zip(*batch)
        return torch.stack(imgs), list(targets)
    

class WheatSegDatasetUnet(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Get all image and mask files
        all_images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        all_masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Only keep images that have corresponding masks
        self.images = []
        self.masks = []
        
        for img_file in all_images:
            # Get the base name without extension
            base_name = os.path.splitext(img_file)[0]
            
            # Look for corresponding mask file with any common extension
            mask_file = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask = base_name + ext
                if potential_mask in all_masks:
                    mask_file = potential_mask
                    break
            
            if mask_file:
                self.images.append(img_file)
                self.masks.append(mask_file)
        
        print(f"Found {len(self.images)} matching image-mask pairs in {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Resize to 256x256
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = image / 255.0
        mask = mask / 255.0

        # Ensure positive strides before converting to tensor
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask

class WheatSegDatasetDETR(Dataset):
    """
    Global Wheat Detection dataset (Kaggle) adapted for DETR.

    Parameters
    ----------
    csv_path : str | Path
        Path to `train.csv` (or any CSV with the five columns:
        `image_id,width,height,bbox,source`).
    images_dir : str | Path
        Directory that contains the images as `<image_id>.jpg`.
    transforms : callable, optional
        A torchvision-style transform applied to the *image* **only**.
        Box coordinates are always kept in absolute pixels and
        converted to normalised cx-cy-w-h *after* transforms.
        If you need box-aware transforms (flip, scale, …) use Albumentations
        or Kornia and wrap the dataset accordingly.
    """

    def __init__(self, csv_path, images_dir, transforms=None):
        self.images_dir = Path(images_dir)
        # Always resize each image to 256 × 256 before any other processing
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),          # converts to float32 in [0, 1]
            ])
        else:
            # Ensure the resize happens first, then run the user‑provided transforms
            if isinstance(transforms, T.Compose):
                self.transforms = T.Compose(
                    [T.Resize((256, 256)), *transforms.transforms]
                )
            else:
                self.transforms = T.Compose(
                    [T.Resize((256, 256)), transforms]
                )

        # ── Build a lookup: image_id → list[xywh] in pixels ───────────────
        df = pd.read_csv(csv_path)
        df["bbox"] = df["bbox"].apply(ast.literal_eval)

        # groupby is ~5× faster than dict(list(zip(...)))
        self.boxes_per_img = (
            df.groupby("image_id")["bbox"]
            .apply(list)
            .to_dict()
        )

        # include images that have *no* boxes
        self.image_ids = sorted([p.stem for p in self.images_dir.glob("*.jpg")])

    # ------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.images_dir / f"{image_id}.jpg"

        pil_img = Image.open(img_path).convert("RGB")
        orig_W, orig_H = pil_img.size           # record original dimensions
        image = self.transforms(pil_img)        # → tensor in [0, 1], 3 × 256 × 256

        # list of [xmin, ymin, w, h] in *pixels*
        xywh_boxes = self.boxes_per_img.get(image_id, [])

        if len(xywh_boxes):
            boxes = torch.as_tensor(xywh_boxes, dtype=torch.float32)
            # convert to cx,cy,w,h and normalise to [0,1]
            boxes[:, 0] += boxes[:, 2] / 2        # x-min + w/2  →  cx
            boxes[:, 1] += boxes[:, 3] / 2        # y-min + h/2  →  cy
            boxes[:, [0, 2]] /= orig_W
            boxes[:, [1, 3]] /= orig_H
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes":  boxes,                           # [N,4] cxcywh in [0,1]
            "labels": torch.zeros(len(boxes), dtype=torch.long),  # single class → 0
            "image_id": image_id,
            "orig_size": torch.tensor([orig_H, orig_W])
        }

        return image, target

    # ------------------------------------------------------------
    # Convenience: batch images, keep targets as list[dict]
    # ------------------------------------------------------------
    @staticmethod
    def collate_fn(batch):
        imgs, targets = tuple(zip(*batch))
        return torch.stack(imgs), list(targets)
