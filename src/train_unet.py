import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb

from unet import UNet
from dataset import WheatSegDatasetUnet


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net model for wheat head segmentation")
    
    # Data arguments
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--masks_dir", type=str, required=True,
                        help="Directory containing training masks")
    parser.add_argument("--val_images_dir", type=str, required=True,
                        help="Directory containing validation images")
    parser.add_argument("--val_masks_dir", type=str, required=True,
                        help="Directory containing validation masks")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Model arguments
    parser.add_argument("--features", nargs='+', type=int, default=[64, 128, 256, 512],
                        help="Feature dimensions for U-Net encoder")
    
    # Logging arguments
    parser.add_argument("--project_name", type=str, default="wheat-detection-unet",
                        help="W&B project name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_unet",
                        help="Directory to save model checkpoints")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Select device
    device = torch.device("mps") if torch.backends.mps.is_available() else \
             torch.device("cuda") if torch.cuda.is_available() else \
             torch.device("cpu")
    
    print(f"=== Training Configuration ===")
    print(f"Training Images: {args.images_dir}")
    print(f"Training Masks: {args.masks_dir}")
    print(f"Validation Images: {args.val_images_dir}")
    print(f"Validation Masks: {args.val_masks_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Device: {device}")
    
    # Create datasets
    train_dataset = WheatSegDatasetUnet(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir
    )
    
    val_dataset = WheatSegDatasetUnet(
        images_dir=args.val_images_dir,
        masks_dir=args.val_masks_dir
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"=== Dataset Summary ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    model = UNet(
        in_channels=3,
        features=args.features,
        lr=args.learning_rate
    )
    
    print(f"=== Model Summary ===")
    print(f"Model: U-Net")
    print(f"Features: {args.features}")
    
    # Setup callbacks and logger
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="unet-{epoch:02d}-{val_ap:.3f}",
        monitor="val_ap",
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
