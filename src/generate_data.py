import numpy as np
import pandas as pd
import os
from PIL import Image
import ast
import random
import glob
import shutil
import tqdm
import argparse

def make_mask_for_image(image_id, annotations_df, output_dir, H=1024, W=1024):
    """
    Generate and save a binary mask for one image_id, based on bounding-box annotations.
    Masks are saved as PNGs with 255 for foreground and 0 for background.
    """
    df_img = annotations_df[annotations_df.image_id == image_id]
    mask = np.zeros((H, W), dtype=np.uint8)
    for _, row in df_img.iterrows():
        # row['bbox'] is a string like "[x, y, width, height]"
        x, y, w, h = ast.literal_eval(row['bbox'])
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + int(w), y1 + int(h)
        mask[y1:y2, x1:x2] = 1
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(mask * 255).save(os.path.join(output_dir, f"{image_id}.png"))

def generate_all_masks(csv_path, output_dir):
    """
    Read the CSV at csv_path and generate masks for every unique image_id,
    storing them in output_dir.
    """
    annotations_df = pd.read_csv(csv_path)
    image_ids = annotations_df.image_id.unique()
    print(f"Generating masks for {len(image_ids)} images...")
    for image_id in tqdm.tqdm(image_ids, desc="Generating masks"):
        make_mask_for_image(image_id, annotations_df, output_dir)


def split_dataset(
    images_dir: str,
    masks_dir: str,
    train_images_dir: str,
    train_masks_dir: str,
    val_images_dir: str,
    val_masks_dir: str,
    unannotated_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    prefix: str = "img"
):
    """
    Splits a dataset into train/val and moves unannotated images aside.
    - Moves images without masks to `unannotated_dir`.
    - Splits the rest into train/val by `val_ratio`.
    - Renames images and masks with `prefix` + counter.
    """
    # Create output directories
    for d in (train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, unannotated_dir):
        os.makedirs(d, exist_ok=True)

    # Gather images and separate unannotated
    all_images = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    annotated = []
    for img_path in all_images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, f"{name}.png")
        if os.path.exists(mask_path):
            annotated.append(img_path)
        else:
            print(f"Image {img_path} has no mask, moving to unannotated directory.")
            shutil.move(img_path, os.path.join(unannotated_dir, os.path.basename(img_path)))

    random.seed(seed)
    random.shuffle(annotated)
    val_count = int(len(annotated) * val_ratio)
    val_set = annotated[:val_count]
    train_set = annotated[val_count:]

    # Move and rename train images/masks
    for i, img_path in enumerate(train_set, start=1):
        base_name = f"{prefix}_{i:04d}"
        dst_img = os.path.join(train_images_dir, base_name + os.path.splitext(img_path)[1])
        shutil.move(img_path, dst_img)
        src_mask = os.path.join(masks_dir, os.path.splitext(os.path.basename(img_path))[0] + ".png")
        dst_mask = os.path.join(train_masks_dir, base_name + ".png")
        shutil.move(src_mask, dst_mask)

    # Move and rename val images/masks
    for i, img_path in enumerate(val_set, start=1):
        base_name = f"{prefix}_{i:04d}"
        dst_img = os.path.join(val_images_dir, base_name + os.path.splitext(img_path)[1])
        shutil.move(img_path, dst_img)
        src_mask = os.path.join(masks_dir, os.path.splitext(os.path.basename(img_path))[0] + ".png")
        dst_mask = os.path.join(val_masks_dir, base_name + ".png")
        shutil.move(src_mask, dst_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images: generate masks and split dataset")
    parser.add_argument("--csv", help="Path to annotations CSV for mask generation")
    parser.add_argument("--images_dir", help="Directory of input images")
    parser.add_argument("--masks_dir", help="Directory of input masks")
    parser.add_argument("--output_masks_dir", help="Where to save generated masks")
    parser.add_argument("--train_images_dir", help="Output directory for training images", required=True)
    parser.add_argument("--train_masks_dir", help="Output directory for training masks", required=True)
    parser.add_argument("--val_images_dir", help="Output directory for validation images", required=True)
    parser.add_argument("--val_masks_dir", help="Output directory for validation masks", required=True)
    parser.add_argument("--unannotated_dir", help="Directory for unannotated images", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prefix", default="img", help="Filename prefix for renamed images")
    args = parser.parse_args()

    # Generate masks if requested
    if args.csv and args.output_masks_dir:
        generate_all_masks(csv_path=args.csv, output_dir=args.output_masks_dir)
        masks_src = args.output_masks_dir
    else:
        masks_src = args.masks_dir

    # Split and rename dataset
    split_dataset(
        images_dir=args.images_dir,
        masks_dir=masks_src,
        train_images_dir=args.train_images_dir,
        train_masks_dir=args.train_masks_dir,
        val_images_dir=args.val_images_dir,
        val_masks_dir=args.val_masks_dir,
        unannotated_dir=args.unannotated_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        prefix=args.prefix
    )
