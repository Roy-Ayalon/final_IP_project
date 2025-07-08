import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class DETR(nn.Module):
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
        backbone=None
    ):
        super().__init__()
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
        hs = self.transformer(src + pos, tgt + query_embed)  # [num_decoder_layers, num_queries, B, hidden_dim]
        hs = hs[-1].permute(1, 0, 2)  # [B, num_queries, hidden_dim]

        # Predict classes and boxes
        logits = self.class_embed(hs)                      # [B, num_queries, num_classes+1]
        boxes = self.bbox_embed(hs).sigmoid()              # [B, num_queries, 4]

        return {'pred_logits': logits, 'pred_boxes': boxes}
