import numpy as np
import pandas as pd
import os
from PIL import Image
import ast
import random
import glob
import shutil
import tqdm

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
    images_dir: str = "data/train",
    masks_dir: str = "data/train_masks",
    val_images_dir: str = "data/val",
    val_masks_dir: str = "data/val_masks",
    val_ratio: float = 0.2,
    seed: int = 42
):
    """
    Splits a fraction of the dataset into a validation set.
    Moves val_ratio of images and their masks from images_dir/masks_dir
    into val_images_dir/val_masks_dir. Remaining files stay in the original train dirs.
    """
    # Create validation directories if they don't exist
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)

    # List all image files and filter to those with masks
    all_images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    images = []
    for img_path in all_images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, f"{name}.png")
        if os.path.exists(mask_path):
            images.append(img_path)
        else:
            print(f"Skipping unannotated image {name}.jpg")
    random.seed(seed)
    random.shuffle(images)

        # Compute number of validation samples
    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]

    print(f"Annotated images: {len(images)}, moving {val_count} to validation set")

    # Move images and corresponding masks to validation directories
    for img_path in tqdm.tqdm(val_images, desc="Moving images to validation set"):
        base = os.path.basename(img_path)
        name, _ = os.path.splitext(base)
        mask_path = os.path.join(masks_dir, f"{name}.png")

        # Move image into validation
        dst_img = os.path.join(val_images_dir, base)
        shutil.move(img_path, dst_img)

        # Move corresponding mask if it exists
        if os.path.exists(mask_path):
            dst_mask = os.path.join(val_masks_dir, f"{name}.png")
            shutil.move(mask_path, dst_mask)
        else:
            print(f"Warning: mask not found for {base}")

    print("Validation split complete.")

if __name__ == "__main__":
    generate_all_masks(
        csv_path="data/train.csv",
        output_dir="data/train_masks"
    )
    # Run splitting with default parameters
    split_dataset()
