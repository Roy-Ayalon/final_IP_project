# %% Cell 1: Imports and Device Setup
import time
import torch
from torch import nn, optim
from torchmetrics import JaccardIndex
from dataset import get_dataloaders
from unet import UNet

if __name__ == "__main__":
    # Device: MPS if available (macOS), otherwise CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # %% Cell 2: DataLoaders
    # Hyperparameters
    BATCH_SIZE = 4
    NUM_WORKERS = 2

    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # %% Cell 3: Model, Loss, Optimizer
    model = UNet().to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # %% Cell 4: Training and Validation Loop
    NUM_EPOCHS = 20
    history = {"train_loss": [], "val_iou": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        # ---- Training ----
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)  # shape [B,1,H,W]
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # ---- Validation ----
        model.eval()
        metric = JaccardIndex(task="binary").to(device)
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                preds = (model(imgs) > 0.5).float()
                metric.update(preds, masks)
        val_iou = metric.compute().item()
        history["val_iou"].append(val_iou)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{NUM_EPOCHS} — "
              f"Train Loss: {train_loss:.4f}, "
              f"Val IoU: {val_iou:.4f}, "
              f"Time: {elapsed:.1f}s")

    # %% Cell 5: Save Model and Plot Results
    # Save the trained model weights
    torch.save(model.state_dict(), "unet_baseline_mps.pth")
    print("Saved model to unet_baseline_mps.pth")

    # Plot training curves
    import matplotlib.pyplot as plt

    epochs = list(range(1, NUM_EPOCHS + 1))
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], '-o')
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, history["val_iou"], '-o')
    plt.title("Validation IoU")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.tight_layout()
    plt.show()