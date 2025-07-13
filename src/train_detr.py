import time
import torch
from torch import nn, optim
from tqdm import tqdm
from dataset import WheatSegDatasetDETR, WheatSegDatasetDETRMultiAug
from Detr import DETR
import cv2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader, Subset

from definitions import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train DETR model for wheat head detection")
    
    # Data arguments
    parser.add_argument("--csv_path", type=str, default="data/train.csv",
                        help="Path to CSV annotations file")
    parser.add_argument("--images_dir", type=str, default="data/train",
                        help="Directory containing training images")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation split ratio")
    
    # Model arguments
    parser.add_argument("--num_queries", type=int, default=100,
                        help="Number of object queries for DETR")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension size")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs")
    
    # Logging arguments
    parser.add_argument("--project_name", type=str, default="wheat-detection-detr",
                        help="W&B project name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    
    return parser.parse_args()

# Select device
device = torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("cpu")

print(f"Using device: {device}")


def main():
    args = parse_args()
    
    print(f"=== Training Configuration ===")
    print(f"CSV Path: {args.csv_path}")
    print(f"Images Directory: {args.images_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Device: {device}")
    print(f"Validation Ratio: {args.val_ratio}")
    
    # Create dataset
    full_dataset = WheatSegDatasetDETRMultiAug(csv_path=args.csv_path, images_dir=args.images_dir)
    
    # Split to Train and Validation
    np.random.seed(42)  # For reproducible splits
    val_indices = np.random.choice(
        len(full_dataset),
        size=int(len(full_dataset) * args.val_ratio),
        replace=False
    )
    val_dataset = Subset(full_dataset, val_indices)
    train_indices = list(set(range(len(full_dataset))) - set(val_indices))
    train_dataset = Subset(full_dataset, train_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=full_dataset.collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=full_dataset.collate_fn,
        pin_memory=True
    )

    print(f"=== Dataset Summary ===")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    model = DETR(
        num_classes=1,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        lr=args.learning_rate,
        warmup_epochs=args.warmup_epochs
    ).to(device)

    print(f"=== Model Summary ===")
    print(f"Model: DETR")
    print(f"Number of queries: {args.num_queries}")
    print(f"Hidden dimension: {args.hidden_dim}")

    # Setup callbacks and logger
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="detr-{epoch:02d}-{val_map:.3f}",
        monitor="val_map",
        mode="max",
        save_top_k=3,
        every_n_epochs=1,
    )
    
    wandb_logger = WandbLogger(project=args.project_name, log_model="all")
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[ckpt_cb],
        logger=wandb_logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    print(f"=== Starting Training ===")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    wandb.finish()
    print(f"=== Training Completed ===")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
