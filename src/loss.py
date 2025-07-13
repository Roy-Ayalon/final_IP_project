from torchvision.ops import generalized_box_iou, box_convert
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F


class HungarianSetCriterion1C(nn.Module):
    """
    Hungarian matching + set loss for single-class detection.
    """
    def __init__(self, eos_coef=0.1, l1=5.0, giou=2.0):
        super().__init__()
        self.eos_coef = eos_coef
        self.l1_w, self.giou_w = l1, giou
        self.register_buffer("ce_weight", torch.tensor([1.0, eos_coef]))

    # Hungarian matching
    @torch.no_grad()
    def _match(self, p_logits, p_boxes, tgt_boxes):
        # Convert to probability and build the cost
        prob_fg   = p_logits.softmax(-1)[:,0]
        
        cost_class = -prob_fg
        # L1 + GIoU costs
        cost_l1   = torch.cdist(p_boxes, tgt_boxes, p=1)  # [Nq, Nt]
        giou      = generalized_box_iou(box_convert(p_boxes, 'cxcywh', 'xyxy'), box_convert(tgt_boxes, 'cxcywh', 'xyxy'))
        cost_giou = 1 - giou

        C = cost_class[:, None] + self.l1_w * cost_l1 + self.giou_w * cost_giou
        idx_q, idx_t = linear_sum_assignment(C.cpu())
        return torch.as_tensor(idx_q), torch.as_tensor(idx_t)

    def forward(self, outputs, targets):
        p_logits, p_boxes = outputs["pred_logits"], outputs["pred_boxes"]
        bs, Nq, _ = p_logits.shape
        loss_cls = loss_l1 = loss_giou = 0.0

        for b in range(bs):
            tgt_boxes = targets[b]["boxes"]
            if tgt_boxes.numel() == 0:
                tgt_full = p_logits.new_full((Nq,), 1, dtype=torch.long)  
                loss_cls += F.cross_entropy(p_logits[b], tgt_full, weight=self.ce_weight,)
                continue
            idx_q, idx_t = self._match(p_logits[b], p_boxes[b], tgt_boxes)
            num_match = len(idx_q)
            tgt_full = p_logits.new_full((Nq,), 1, dtype=torch.long)  
            tgt_full[idx_q] = 0                                      

            loss_cls += F.cross_entropy(p_logits[b], tgt_full, weight=self.ce_weight)
            matched_p = p_boxes[b, idx_q]
            matched_t = tgt_boxes[idx_t]
            loss_l1   += F.l1_loss(matched_p, matched_t, reduction="sum") / max(num_match, 1)
            loss_giou += (1 - generalized_box_iou(box_convert(matched_p, 'cxcywh', 'xyxy'), box_convert(matched_t, 'cxcywh', 'xyxy')).diag()).sum() / max(num_match, 1)
            

        return loss_cls + self.l1_w * loss_l1 + self.giou_w * loss_giou