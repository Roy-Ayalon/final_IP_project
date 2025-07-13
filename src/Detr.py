import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes, make_grid
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import math
from loss import HungarianSetCriterion1C
from torchvision.ops import box_convert
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


CONFIDENCE_THRESHOLD_DEFAULT = 0.3
COCO_PRETRAINED_URL = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"


class MLP(nn.Module):
    """
    Simple feed-forward network (MLP) with configurable layers and ReLU activations.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            output_dim = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PositionEmbeddingSine(nn.Module):
    """
    Sine-cosine positional encoding as in the DETR paper.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None and normalize:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DETR(pl.LightningModule):
    """
    DETR model: backbone + transformer + prediction heads.
    Outputs raw logits and normalized bounding boxes.
    """
    def __init__(
        self,
        num_classes,
        num_queries=100,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        backbone=None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
        pretrained: bool = False,
        pretrained_url: str = COCO_PRETRAINED_URL,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        # Backbone
        if backbone is None:
            resnet = torch.hub.load(
                'pytorch/vision:v0.10.0', 'resnet50', pretrained=True
            )
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.backbone_out_channels = 2048
        else:
            self.backbone = backbone
            self.backbone_out_channels = backbone.num_out_channels

        # Input projection to transformer dimension
        self.input_proj = nn.Conv2d(
            self.backbone_out_channels, hidden_dim, kernel_size=1
        )

        # Positional encoding
        self.position_encoding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self.val_map_metric = MeanAveragePrecision(
            box_format="cxcywh",   # our boxes are in (cx,cy,w,h) normalised
            iou_type="bbox",
        )
        self.train_map_metric = MeanAveragePrecision(
            box_format="cxcywh",
            iou_type="bbox",
        )
        
        self.criterion = HungarianSetCriterion1C(eos_coef=0.1)

        # Lightning‑specific hyper‑parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.confidence_threshold = confidence_threshold
        self.warmup_epochs = warmup_epochs
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(pretrained_url, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # x: [B, 3, H, W]
        bs, _, H, W = x.shape
        # Backbone feature map
        feat = self.backbone(x)  # [B, C_backbone, H', W']
        # Project features
        proj = self.input_proj(feat)  # [B, hidden_dim, H', W']
        # Positional encoding
        pos = self.position_encoding(proj)

        # Flatten spatial dims
        src = proj.flatten(2).permute(2, 0, 1)  # [H'*W', B, hidden_dim]
        pos = pos.flatten(2).permute(2, 0, 1)   # [H'*W', B, hidden_dim]

        # Prepare queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        # Transformer forward
        hs = self.transformer(src + pos, tgt + query_embed)
        hs = hs.permute(1, 0, 2)  # [B, num_queries, hidden_dim]
        

        # Predict classes and boxes
        logits = self.class_embed(hs)                      # [B, num_queries, num_classes+1]
        boxes = self.bbox_embed(hs).sigmoid()              # [B, num_queries, 4]

        return {'pred_logits': logits, 'pred_boxes': boxes}

    # ---------------------------------------------------------------------
    # Lightning helpers
    # ---------------------------------------------------------------------
    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.compute_loss(outputs, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # ────────────────────────────────────────────────
        # mAP (train) update
        # ────────────────────────────────────────────────
        preds, gts = [], []
        Nq = outputs["pred_logits"].shape[1]

        for i in range(images.size(0)):
            logits = outputs["pred_logits"][i]
            probs  = logits.softmax(-1)
            scores, labels = probs[:, :-1].max(-1)
            boxes = outputs["pred_boxes"][i]

            keep = torch.arange(Nq, device=boxes.device)  # keep all queries

            preds.append({
                "boxes":  boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            })

            gts.append({
                "boxes":  targets[i]["boxes"],
                "labels": targets[i].get(
                    "labels",
                    torch.zeros(len(targets[i]["boxes"]),
                                dtype=torch.long,
                                device=boxes.device)
                ),
            })

        self.train_map_metric.update(preds, gts)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.compute_loss(outputs, targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        preds, gts = [], []
        Nq = outputs["pred_logits"].shape[1]

        for i in range(images.size(0)):
            logits = outputs["pred_logits"][i]           # [Nq, 1+ε]
            probs  = logits.softmax(-1)                  # convert to probabilities

            # Foreground (non‑background) confidence & label
            scores, labels = probs[:, :-1].max(-1)       # skip ε / background column
            boxes   = outputs["pred_boxes"][i]           # [Nq,4]

            keep = torch.arange(Nq) # Keep all queries

            preds.append({
                "boxes":  boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],                  # single fg class → all zeros
            })

            # ­­­ Ground-truth
            gts.append({
                "boxes":  targets[i]["boxes"],
                "labels": targets[i].get(
                    "labels",
                    torch.zeros(len(targets[i]["boxes"]),
                                dtype=torch.long,
                                device=boxes.device)
                ),
            })

        self.val_map_metric.update(preds, gts)
        # cache the FIRST image of the FIRST batch once per epoch for media logging
        if batch_idx == 0:
            random_idx = torch.randint(0, images.size(0), (1,)).item()
            self._val_vis_image = images[random_idx].detach().cpu()
            self._val_vis_pred_logits = outputs["pred_logits"][random_idx].detach().cpu()
            self._val_vis_pred_boxes  = outputs["pred_boxes"][random_idx].detach().cpu()
            self._val_vis_target_boxes = targets[random_idx]["boxes"].detach().cpu()
        return loss

    # -----------------------------------------------------------------
    # Media logging
    # -----------------------------------------------------------------
    @staticmethod
    def _xywh_to_xyxy_norm(boxes):
        """
        Convert normalised (cx, cy, w, h) in [0,1] to (x1,y1,x2,y2) also in [0,1].
        """
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def on_validation_epoch_end(self):
        """
        Log one composite image (original + predictions) once per epoch.
        Works automatically with TensorBoardLogger.
        """
        map_res = self.val_map_metric.compute()     # dict with 'map', 'map_50', …
        self.log("val_map", map_res["map"], prog_bar=True, on_epoch=True)
        self.val_map_metric.reset()
        
        if not hasattr(self, "_val_vis_image"):
            return  # nothing cached (e.g. in distributed eval)
        img  = (self._val_vis_image.clone() * 255).byte()      # [3,H,W] uint8

        H, W = img.shape[1:]

        # ------------------------------------------------------------------
        # Convert cached cxcywh ∈ [0,1] → xyxy pixels for both GT & preds
        # ------------------------------------------------------------------
        # ground‑truth
        gt_boxes_xyxy = self._xywh_to_xyxy_norm(
            self._val_vis_target_boxes.clone()
        ) * torch.tensor([W, H, W, H])
        gt_boxes_px = gt_boxes_xyxy.int()

        # predicted boxes: filter by class != 'no‑object' & confidence
        pred_logits = self._val_vis_pred_logits      # [Nq,C+1]
        probs = pred_logits.softmax(-1)
        scores, labels = probs[:, :-1].max(-1)       # foreground class confidence
        keep = scores > self.confidence_threshold    # filter by confidence
        pred_boxes = self._val_vis_pred_boxes[keep]
        pred_boxes_xyxy = self._xywh_to_xyxy_norm(pred_boxes) * torch.tensor([W, H, W, H])
        pred_boxes_px = pred_boxes_xyxy.int()

        # draw
        vis_pred = draw_bounding_boxes(img, pred_boxes_px, colors="red", width=3)
        vis_gt   = draw_bounding_boxes(img, gt_boxes_px,   colors="blue", width=3)

        grid = make_grid([vis_gt, vis_pred], nrow=2)

        # Log image with WandbLogger if available, otherwise fallback to TensorBoard
        if isinstance(self.logger, WandbLogger):
            # Lightning helper logs to W&B media panel
            self.logger.log_image(
                key="val/gt_vs_pred",
                images=[grid],
                step=self.current_epoch,
            )
        elif hasattr(self.logger, "experiment") and callable(getattr(self.logger.experiment, "add_image", None)):
            # TensorBoard fallback
            self.logger.experiment.add_image("val/gt_vs_pred", grid, global_step=self.current_epoch)

    def on_train_epoch_end(self):
        """
        Compute & log mAP over the entire training epoch, then reset.
        """
        map_res = self.train_map_metric.compute()
        self.log("train_map", map_res["map"], prog_bar=True, on_epoch=True)
        self.train_map_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Optional linear warm‑up over the first `warmup_epochs` epochs.
        if self.warmup_epochs > 0:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: min((epoch + 1) / float(self.warmup_epochs), 1.0),
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

        return optimizer
