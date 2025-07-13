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
import zipfile
import subprocess
import sys

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

def download_kaggle_dataset(username=None, key=None):
    """
    Download and extract the Global Wheat Detection dataset from Kaggle.
    Returns the paths to the extracted train images and CSV file.
    """
    try:
        # Check if kaggle is installed
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Kaggle CLI not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
    
    print("🔍 Setting up Kaggle API credentials...")
    
    # Set up environment variables for Kaggle API if provided via CLI
    if username and key:
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        print(f"✅ Using provided API credentials for user: {username}")
    else:
        # Check for existing credentials
        kaggle_dir = os.path.expanduser("~/.kaggle")
        kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
        
        if not os.path.exists(kaggle_json) and not (os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY')):
            print("❌ Kaggle API credentials not found!")
            print("Please provide credentials in one of these ways:")
            print("1. Use --kaggle_username and --kaggle_key arguments")
            print("2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
            print("3. Create ~/.kaggle/kaggle.json file:")
            print("   - Go to https://www.kaggle.com/account")
            print("   - Click 'Create New API Token' to download kaggle.json")
            print("   - Move kaggle.json to ~/.kaggle/kaggle.json")
            print("   - Run: chmod 600 ~/.kaggle/kaggle.json")
            sys.exit(1)
        print("✅ Using existing Kaggle credentials")
    
    # Create data directory
    data_dir = "wheat_data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("📥 Downloading Global Wheat Detection dataset from Kaggle...")
    
    # Download the dataset
    try:
        subprocess.run([
            "kaggle", "competitions", "download", 
            "-c", "global-wheat-detection", 
            "-p", data_dir
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download dataset: {e}")
        print("Make sure you have accepted the competition rules at:")
        print("https://www.kaggle.com/c/global-wheat-detection/rules")
        sys.exit(1)
    
    print("📦 Extracting dataset...")
    
    # Extract all zip files
    for zip_file in glob.glob(os.path.join(data_dir, "*.zip")):
        print(f"Extracting {os.path.basename(zip_file)}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_file)  # Clean up zip file
    
    # Find the train directory and CSV file
    train_dir = os.path.join(data_dir, "train")
    csv_path = os.path.join(data_dir, "train.csv")
    
    if not os.path.exists(train_dir):
        print(f"❌ Train directory not found at {train_dir}")
        sys.exit(1)
    
    if not os.path.exists(csv_path):
        print(f"❌ Train CSV not found at {csv_path}")
        sys.exit(1)
    
    print(f"✅ Dataset downloaded and extracted to {data_dir}")
    print(f"📁 Images directory: {train_dir}")
    print(f"📄 CSV file: {csv_path}")
    
    return train_dir, csv_path


def setup_unet_data(csv_path, images_dir, val_ratio=0.2, seed=42):
    """
    Setup data structure for U-Net training:
    - Generate masks from CSV annotations
    - Split into train/val sets
    - Organize into proper folder structure
    """
    print("🏗️ Setting up data for U-Net training...")
    
    # Create folder structure
    base_dir = "data_unet"
    os.makedirs(base_dir, exist_ok=True)
    
    masks_dir = os.path.join(base_dir, "masks")
    train_images_dir = os.path.join(base_dir, "train", "images")
    train_masks_dir = os.path.join(base_dir, "train", "masks")
    val_images_dir = os.path.join(base_dir, "val", "images")
    val_masks_dir = os.path.join(base_dir, "val", "masks")
    unannotated_dir = os.path.join(base_dir, "unannotated")
    
    # Generate masks from CSV
    print("🎭 Generating masks from CSV annotations...")
    generate_all_masks(csv_path=csv_path, output_dir=masks_dir)
    
    # Split dataset
    print("✂️ Splitting dataset into train/val...")
    split_dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        train_images_dir=train_images_dir,
        train_masks_dir=train_masks_dir,
        val_images_dir=val_images_dir,
        val_masks_dir=val_masks_dir,
        unannotated_dir=unannotated_dir,
        val_ratio=val_ratio,
        seed=seed,
        prefix="img"
    )
    
    print(f"✅ U-Net data prepared in '{base_dir}' folder")
    print(f"🚀 To train U-Net, run:")
    print(f"   python src/train_unet.py --images_dir {train_images_dir} --masks_dir {train_masks_dir} --val_images_dir {val_images_dir} --val_masks_dir {val_masks_dir}")


def setup_detr_data(csv_path, images_dir, val_ratio=0.2, seed=42):
    """
    Setup data structure for DETR training:
    - Copy images to data folder
    - Copy CSV annotations
    - Ready for DETR training
    """
    print("🏗️ Setting up data for DETR training...")
    
    # Create folder structure
    base_dir = "data_detr"
    os.makedirs(base_dir, exist_ok=True)
    
    detr_images_dir = os.path.join(base_dir, "train")
    os.makedirs(detr_images_dir, exist_ok=True)
    
    # Copy all images to DETR data folder
    print("📁 Copying images to DETR data folder...")
    all_images = glob.glob(os.path.join(images_dir, "*.*"))
    for img_path in tqdm.tqdm(all_images, desc="Copying images"):
        shutil.copy2(img_path, os.path.join(detr_images_dir, os.path.basename(img_path)))
    
    # Copy CSV file
    detr_csv_path = os.path.join(base_dir, "train.csv")
    shutil.copy2(csv_path, detr_csv_path)
    
    print(f"✅ DETR data prepared in '{base_dir}' folder")
    print(f"🚀 To train DETR, run:")
    print(f"   python src/train_detr.py --csv_path {detr_csv_path} --images_dir {detr_images_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Kaggle wheat dataset and prepare for U-Net or DETR training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data and setup for U-Net training (with API credentials)
  python generate_data.py --model unet --kaggle_username YOUR_USERNAME --kaggle_key YOUR_API_KEY

  # Download data and setup for DETR training  
  python generate_data.py --model detr --kaggle_username YOUR_USERNAME --kaggle_key YOUR_API_KEY

  # Use existing kaggle.json file or environment variables
  python generate_data.py --model unet

  # Use custom validation ratio
  python generate_data.py --model unet --val_ratio 0.15 --kaggle_username YOUR_USERNAME --kaggle_key YOUR_API_KEY
        """
    )
    parser.add_argument("--model", choices=["unet", "detr"], required=True,
                        help="Model type to prepare data for")
    parser.add_argument("--kaggle_username", type=str,
                        help="Kaggle username for API access")
    parser.add_argument("--kaggle_key", type=str,
                        help="Kaggle API key for access")
    parser.add_argument("--val_ratio", type=float, default=0.2, 
                        help="Validation set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download and use existing data in wheat_data/")
    args = parser.parse_args()

    # Download dataset from Kaggle (unless skipping)
    if args.skip_download:
        print("⏭️ Skipping download, using existing data...")
        train_dir = "wheat_data/train"
        csv_path = "wheat_data/train.csv"
        
        if not os.path.exists(train_dir) or not os.path.exists(csv_path):
            print("❌ Existing data not found! Remove --skip_download to download fresh data.")
            sys.exit(1)
    else:
        train_dir, csv_path = download_kaggle_dataset(
            username=args.kaggle_username,
            key=args.kaggle_key
        )

    # Setup data for the specified model
    if args.model == "unet":
        setup_unet_data(
            csv_path=csv_path,
            images_dir=train_dir,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
    elif args.model == "detr":
        setup_detr_data(
            csv_path=csv_path,
            images_dir=train_dir,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
    
    print("\n🎉 Data setup complete! You can now run the training command shown above.")
