import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
import torchmetrics

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(pl.LightningModule):
    def __init__(self, in_channels=3, features=[64,128,256,512], lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        # Average Precision metric for segmentation masks
        self.val_ap = torchmetrics.AveragePrecision(task='binary').to("cpu")
        print(f"Using device {self.val_ap.device} for AveragePrecision metric")
        # encoder
        self.downs = nn.ModuleList()
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f
        self.pool = nn.MaxPool2d(2)
        # bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # decoder
        self.ups = nn.ModuleList()
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f*2, f))
        # final conv
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[-(idx//2 + 1)]
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx+1](x)
        return torch.sigmoid(self.final_conv(x) / 0.1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.binary_cross_entropy(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.binary_cross_entropy(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.val_ap.update(preds.detach().cpu(), y.int().detach().cpu())
        if batch_idx == 0:
            self._val_img = x[0].detach().cpu()
            self._val_gt  = y[0].detach().cpu()
            self._val_pred = preds[0].detach().cpu()
        return loss

    def on_validation_epoch_end(self):
        """
        Log a 3-panel image: original, ground-truth mask, and predicted mask.
        """
        ap = self.val_ap.compute().item()
        print(f"AP: {ap}")
        self.log("val_ap", ap, on_epoch=True, prog_bar=True)
        self.val_ap.reset()
        
        if not hasattr(self, "_val_img"):
            return

        # prepare images: [C,H,W], float32 in [0,1]
        orig = self._val_img                       # [3,H,W]
        gt   = self._val_gt                        # [1,H,W]
        pred = (self._val_pred > 0.5).float()      # [1,H,W]

        # convert single-channel to 3-channel
        def _to_3ch(img):
            return img.repeat(3, 1, 1) if img.shape[0] == 1 else img

        orig3 = _to_3ch(orig)
        gt3   = _to_3ch(gt)
        pred3 = _to_3ch(pred)

        # build grid
        grid = make_grid([orig3, gt3, pred3], nrow=3)

        # log to WandB if available, else TensorBoard
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key="val/unet_gt_pred",
                images=[grid],
                step=self.current_epoch,
            )
        elif hasattr(self.logger, "experiment") and callable(getattr(self.logger.experiment, "add_image", None)):
            self.logger.experiment.add_image("val/unet_gt_pred", grid, global_step=self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)