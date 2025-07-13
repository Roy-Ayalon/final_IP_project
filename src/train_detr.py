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
from torch.utils.data import DataLoader, Subset
from segmentation_models_pytorch.losses import TverskyLoss
# Import TverskyLoss from torch

from utils import *
from definitions import *

# Select MPS if available, otherwise CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
DETR_MODEL_SAVE_PATH = "detr.pth"
CSV_PATH = "/gpfs0/bgu-benshimo/users/guyperet/final_IP_project/data_detr/train.csv"
IMAGES_DIR_PATH = "/gpfs0/bgu-benshimo/users/guyperet/final_IP_project/data_detr/train"


def main():
    full_dataset = WheatSegDatasetDETR(csv_path=CSV_PATH, images_dir=IMAGES_DIR_PATH)
    # Split to Train and Validation using VAL_RATIO
    val_indices = np.random.choice(
        len(full_dataset),
        size=int(len(full_dataset) * VAL_RATIO),
        replace=False
    )
    val_dataset = Subset(full_dataset, val_indices)
    train_indices = list(set(range(len(full_dataset))) - set(val_indices))
    train_dataset = Subset(full_dataset, train_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=full_dataset.collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=full_dataset.collate_fn,
        pin_memory=True
    )

    print(f"number of training samples: {len(train_loader.dataset)}")
    print(f"number of validation samples: {len(val_loader.dataset)}")
    print(f"dataloaders created with batch size {BATCH_SIZE} and {NUM_WORKERS} workers")
    print(f"=== Dataloaders Summary ===")
    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Validation Loader: {len(val_loader)} batches")


    model   = DETR(num_classes=1).to(device)

    print(f"=== Model Summary ===")
    print(model)

    ckpt_cb = ModelCheckpoint(  dirpath="checkpoints",
                                filename="detr-{epoch:02d}-{val_map:.3f}",
                                monitor="val_map",
                                mode="max",
                                save_top_k=3,
                                every_n_epochs=1,)
    wandb_logger = WandbLogger(project="wheat-detection-detr", log_model="all")
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        callbacks=[ckpt_cb],
        logger=wandb_logger,
        enable_checkpointing= True,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    wandb.finish()
    print(f"=== Training Completed ===")
    
    
if __name__ == "__main__":
    main()
