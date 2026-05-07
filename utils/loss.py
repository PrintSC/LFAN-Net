# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.modules.fan import (
    build_box_prior,
    build_rfft_band_masks,
    make_edge_mask,
    masked_rfft,
    resize_prior,
    split_guidance_masks,
    wrapped_phase_l1,
)
from utils.metrics import OKS_SIGMA
from utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing
    on hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if sample_weight is not None:
            weight = weight * sample_weight
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if sample_weight is not None:
            weight = weight * sample_weight
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.fan_cls_balance = float(getattr(h, "fan_cls_balance", 0.0))
        self.fan_cls_balance_pow = float(getattr(h, "fan_cls_balance_pow", 0.5))
        self.fan_cls_balance_cap = float(getattr(h, "fan_cls_balance_cap", 4.0))
        self.fan_small_box = float(getattr(h, "fan_small_box", 0.0))
        self.fan_small_area_ref = float(getattr(h, "fan_small_area_ref", 0.005))
        self.fan_small_box_max = float(getattr(h, "fan_small_box_max", 3.0))
        self.fan_bg_suppress = float(getattr(h, "fan_bg_suppress", 0.0))
        self.fan_bg_topk = float(getattr(h, "fan_bg_topk", 0.005))
        self.fan_bg_gamma = float(getattr(h, "fan_bg_gamma", 1.5))
        self.fan_bg_warmup = max(int(getattr(h, "fan_bg_warmup", 12000) or 12000), 1)
        self.fan_bg_rare_floor = float(getattr(h, "fan_bg_rare_floor", 0.25))
        self.fan_bg_rare_pow = float(getattr(h, "fan_bg_rare_pow", 0.5))
        self.fan_det = float(getattr(h, "fan_det", 0.0))
        self.fan_det_margin = float(getattr(h, "fan_det_margin", 0.05))
        self.fan_det_warmup = max(int(getattr(h, "fan_det_warmup", 4000) or 4000), 1)
        self.bg_suppress_steps = 0
        self.det_steps = 0

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute a stable masked mean for dense box-guided regularization."""
        mask = mask.to(dtype=x.dtype)
        return (x * mask).sum() / mask.sum().clamp_min(1e-6)

    def _build_cls_balance_scale(
        self, batch: Dict[str, torch.Tensor], target_scores: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Upweight positive classification targets from rare classes within the current batch."""
        if self.fan_cls_balance <= 0 or batch["cls"].numel() == 0:
            return torch.ones_like(target_scores, dtype=dtype)

        cls_ids = batch["cls"].view(-1).to(device=target_scores.device, dtype=torch.long)
        counts = torch.bincount(cls_ids, minlength=self.nc).to(device=target_scores.device, dtype=torch.float32)
        present = counts > 0
        if not present.any():
            return torch.ones_like(target_scores, dtype=dtype)

        inv = counts.sum().clamp_min(1.0) / counts.clamp_min(1.0)
        inv = inv.pow(self.fan_cls_balance_pow)
        inv = inv / inv[present].mean().clamp_min(1e-6)
        inv = inv.clamp(1.0, self.fan_cls_balance_cap)
        cls_scale = (1.0 + self.fan_cls_balance * (inv - 1.0)).to(dtype=dtype).view(1, 1, -1)
        return 1.0 + (cls_scale - 1.0) * target_scores.to(dtype)

    def _build_small_box_scale(
        self, target_bboxes: torch.Tensor, target_scores: torch.Tensor, imgsz: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Increase loss weights for tiny assigned boxes to improve recall on small objects."""
        if self.fan_small_box <= 0:
            return torch.ones((*target_scores.shape[:2], 1), device=target_scores.device, dtype=dtype)

        wh = (target_bboxes[..., 2:4] - target_bboxes[..., 0:2]).clamp_min(0.0).float()
        img_hw = imgsz.float()
        img_area = (img_hw[0] * img_hw[1]).clamp_min(1.0)
        box_area = (wh[..., 0] * wh[..., 1]) / img_area
        area_ref = torch.tensor(self.fan_small_area_ref, device=target_scores.device, dtype=box_area.dtype)
        scale = torch.sqrt(area_ref / box_area.clamp_min(1e-6)).clamp(1.0, self.fan_small_box_max)
        scale = 1.0 + self.fan_small_box * (scale - 1.0)
        pos_mask = target_scores.sum(-1, keepdim=True).gt(0).to(dtype=dtype)
        return 1.0 + (scale.unsqueeze(-1).to(dtype=dtype) - 1.0) * pos_mask

    def _build_bg_suppress_scale(
        self,
        feats: List[torch.Tensor],
        pred_scores: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        target_scores: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Upweight clear-background hard negatives using GT box priors to suppress false positives."""
        if self.fan_bg_suppress <= 0 or batch["bboxes"].numel() == 0:
            return torch.ones_like(target_scores, dtype=dtype)

        self.bg_suppress_steps += 1
        suppress_strength = self.fan_bg_suppress * min(self.bg_suppress_steps / self.fan_bg_warmup, 1.0)
        if suppress_strength <= 0:
            return torch.ones_like(target_scores, dtype=dtype)

        batch_size = pred_scores.shape[0]
        level_bg = []
        for feat in feats:
            prior = build_box_prior(
                batch["bboxes"],
                batch["batch_idx"],
                batch_size,
                feat.shape[-2:],
                device=pred_scores.device,
                dtype=dtype,
            )
            _, _, bg = split_guidance_masks(prior, neutral_band=0.15)
            level_bg.append(bg.flatten(2).transpose(1, 2))

        clear_bg = torch.cat(level_bg, dim=1).to(dtype=dtype)
        neg_mask = target_scores.sum(-1, keepdim=True).eq(0).to(dtype=dtype) * clear_bg
        if neg_mask.sum() <= 0:
            return torch.ones_like(target_scores, dtype=dtype)

        pred_prob = pred_scores.detach().sigmoid()
        max_prob = pred_prob.amax(-1, keepdim=True)
        hard_score = max_prob.pow(self.fan_bg_gamma)
        hard_mask = neg_mask

        if 0.0 < self.fan_bg_topk < 1.0:
            topk_mask = torch.zeros_like(hard_mask)
            for b in range(batch_size):
                valid = hard_mask[b, :, 0] > 0
                n_valid = int(valid.sum().item())
                if n_valid == 0:
                    continue
                k = max(1, int(round(n_valid * self.fan_bg_topk)))
                valid_idx = torch.where(valid)[0]
                valid_scores = hard_score[b, valid, 0]
                top_local = valid_scores.topk(min(k, n_valid), sorted=False).indices
                topk_mask[b, valid_idx[top_local], 0] = 1.0
            hard_mask = hard_mask * topk_mask

        class_scale = self._build_bg_class_scale(batch, pred_scores.device, dtype)
        class_hard = pred_prob.pow(self.fan_bg_gamma) * hard_mask * class_scale.view(1, 1, -1)
        return 1.0 + suppress_strength * class_hard

    def _build_bg_class_scale(self, batch: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Reduce background suppression on rare classes so box guidance focuses on dominant false positives."""
        floor = float(min(max(self.fan_bg_rare_floor, 0.0), 1.0))
        if batch["cls"].numel() == 0:
            return torch.ones(self.nc, device=device, dtype=dtype)

        cls_ids = batch["cls"].view(-1).to(device=device, dtype=torch.long)
        counts = torch.bincount(cls_ids, minlength=self.nc).to(device=device, dtype=torch.float32)
        present = counts > 0
        if not present.any():
            return torch.ones(self.nc, device=device, dtype=dtype)

        max_count = counts[present].max().clamp_min(1.0)
        freq = counts / max_count
        if self.fan_bg_rare_pow > 0:
            freq = freq.pow(self.fan_bg_rare_pow)
        else:
            freq = present.to(dtype=freq.dtype)

        scale = torch.full((self.nc,), floor, device=device, dtype=freq.dtype)
        scale[present] = floor + (1.0 - floor) * freq[present]
        return scale.to(dtype=dtype)

    def _box_score_map_loss(self, score_map: torch.Tensor, obj: torch.Tensor, edge: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        """Calibrate dense detection confidence maps with GT box priors without changing the detect head."""
        score_map = score_map.float().clamp(1e-4, 1.0 - 1e-4)
        obj = obj.to(dtype=score_map.dtype)
        edge = edge.to(dtype=score_map.dtype)
        bg = bg.to(dtype=score_map.dtype)
        target = (obj + 0.35 * edge).clamp(0.0, 1.0)
        weights = (0.25 + 2.5 * obj + 1.25 * edge + 0.35 * bg).to(dtype=score_map.dtype)
        with autocast(enabled=False):
            bce = F.binary_cross_entropy(score_map.float(), target.float(), reduction="none")
        bce = (bce.to(dtype=score_map.dtype) * weights).sum() / weights.sum().clamp_min(1e-6)

        obj_score = self._masked_mean(score_map, obj)
        edge_score = self._masked_mean(score_map, edge)
        bg_score = self._masked_mean(score_map, bg)
        loss = 0.50 * bce
        loss = loss + 0.75 * ((1.0 - score_map) * obj).mean()
        loss = loss + 0.35 * ((1.0 - score_map) * edge).mean()
        loss = loss + 1.10 * (score_map * bg).mean()
        loss = loss + 1.25 * F.relu(self.fan_det_margin - obj_score + bg_score)
        loss = loss + 0.75 * F.relu(0.5 * self.fan_det_margin - edge_score + bg_score)
        return loss

    def _build_det_prior_loss(
        self, feats: List[torch.Tensor], pred_scores: torch.Tensor, batch: Dict[str, torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor:
        """Apply GT-box calibration directly on dense class-response maps to suppress false positives."""
        if self.fan_det <= 0 or batch["bboxes"].numel() == 0:
            return torch.zeros((), device=self.device)

        self.det_steps += 1
        det_scale = min(self.det_steps / self.fan_det_warmup, 1.0)
        if det_scale <= 0:
            return torch.zeros((), device=self.device)

        batch_size = pred_scores.shape[0]
        level_start = 0
        total = torch.zeros((), device=self.device)
        level_count = 0
        for feat in feats:
            h, w = feat.shape[-2:]
            n_level = h * w
            level_logits = pred_scores[:, level_start : level_start + n_level, :]
            level_logits = level_logits.transpose(1, 2).reshape(batch_size, self.nc, h, w)
            score_map = level_logits.sigmoid().amax(1, keepdim=True)
            prior = build_box_prior(
                batch["bboxes"],
                batch["batch_idx"],
                batch_size,
                (h, w),
                device=score_map.device,
                dtype=dtype,
            )
            obj, edge, bg = split_guidance_masks(prior, neutral_band=0.10)
            total = total + self._box_score_map_loss(score_map, obj, edge, bg)
            level_start += n_level
            level_count += 1

        return det_scale * self.fan_det * (total / max(level_count, 1))

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        cls_balance = self._build_cls_balance_scale(batch, target_scores, dtype)
        small_box_scale = self._build_small_box_scale(target_bboxes, target_scores, imgsz, dtype)
        bg_suppress_scale = self._build_bg_suppress_scale(feats, pred_scores, batch, target_scores, dtype)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        cls_loss = self.bce(pred_scores, target_scores.to(dtype))
        cls_loss = cls_loss * cls_balance * small_box_scale * bg_suppress_scale
        loss[1] = cls_loss.sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[1] += self._build_det_prior_loss(feats, pred_scores, batch, dtype)
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class FANDetectionLoss(v8DetectionLoss):
    """Detection loss with FAN-specific frequency and task priors."""

    def __init__(self, model):
        super().__init__(model)
        self.prior_bce = nn.BCEWithLogitsLoss()
        self.lambda_freq = float(getattr(model.args, "fan_freq", 0.1))
        self.lambda_task = float(getattr(model.args, "fan_task", 0.15))
        self.lambda_sem = float(getattr(model.args, "fan_sem", 0.0))
        self.w_hf = float(getattr(model.args, "fan_hf", 1.0))
        self.w_lf = float(getattr(model.args, "fan_lf", 1.0))
        self.w_phase = float(getattr(model.args, "fan_phase", 1.0))
        self.w_prior = float(getattr(model.args, "fan_prior", 1.0))
        self.w_gate = float(getattr(model.args, "fan_gate", 1.0))
        self.hf_margin = float(getattr(model.args, "fan_hf_margin", 0.05))
        self.lf_margin = float(getattr(model.args, "fan_lf_margin", 0.02))
        self.obj_bg_margin = float(getattr(model.args, "fan_obj_bg_margin", 0.05))
        self.sem_margin = float(getattr(model.args, "fan_sem_margin", 0.05))
        self.sem_logit = float(getattr(model.args, "fan_sem_logit", 4.0))
        self.loss_size = max(int(getattr(model.args, "fan_loss_size", 48)), 0)
        self.aux_steps = 0
        self.aux_warmup_steps = 1000

    def _compress_map(self, x, mode="avg"):
        """Downsample auxiliary-loss tensors to a fixed spatial budget."""
        if x is None or self.loss_size == 0:
            return x
        h, w = x.shape[-2:]
        size = (min(h, self.loss_size), min(w, self.loss_size))
        if size == (h, w):
            return x
        if mode == "nearest":
            return F.interpolate(x, size=size, mode="nearest")
        if mode == "bilinear":
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if mode == "max":
            return F.adaptive_max_pool2d(x, size)
        if mode == "mix":
            avg = F.adaptive_avg_pool2d(x, size)
            mx = F.adaptive_max_pool2d(x, size)
            return 0.5 * (avg + mx)
        return F.adaptive_avg_pool2d(x, size)

    @staticmethod
    def _masked_mean(x, mask):
        """Compute a stable masked mean."""
        mask = mask.to(dtype=x.dtype)
        return (x * mask).sum() / mask.sum().clamp_min(1e-6)

    @staticmethod
    def _masked_pool(feat, mask):
        """Compute masked feature prototypes."""
        mask = mask.to(dtype=feat.dtype)
        denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (feat * mask).sum(dim=(2, 3), keepdim=True) / denom

    def _map_guidance_loss(self, score_map, obj, edge, bg, target=None):
        """Force a probability-like map to activate inside boxes and suppress background."""
        if score_map is None:
            return torch.zeros((), device=self.device)

        score_map = score_map.clamp(0.0, 1.0)
        obj = resize_prior(obj, score_map.shape[-2:], mode="bilinear", preserve_small=True).to(dtype=score_map.dtype)
        edge = resize_prior(edge, score_map.shape[-2:], mode="bilinear", preserve_small=True).to(dtype=score_map.dtype)
        bg = resize_prior(bg, score_map.shape[-2:], mode="bilinear", preserve_small=True).to(dtype=score_map.dtype)
        if target is None:
            target = (obj + 0.5 * edge).clamp(0.0, 1.0)
        else:
            target = resize_prior(target, score_map.shape[-2:], mode="bilinear", preserve_small=True).to(dtype=score_map.dtype)

        weights = (1.0 + 2.0 * obj + edge).to(dtype=score_map.dtype)
        with autocast(enabled=False):
            bce = F.binary_cross_entropy(
                score_map.float().clamp(1e-4, 1.0 - 1e-4), target.float(), reduction="none"
            )
        bce = (bce.to(dtype=score_map.dtype) * weights).sum() / weights.sum().clamp_min(1e-6)

        obj_score = self._masked_mean(score_map, obj)
        edge_score = self._masked_mean(score_map, edge)
        bg_score = self._masked_mean(score_map, bg)
        loss = ((1.0 - score_map) * obj).mean()
        loss = loss + 0.75 * ((1.0 - score_map) * edge).mean()
        loss = loss + (score_map * bg).mean()
        loss = loss + 0.5 * bce
        loss = loss + 1.5 * F.relu(self.obj_bg_margin - obj_score + bg_score)
        loss = loss + 1.0 * F.relu(0.5 * self.obj_bg_margin - edge_score + bg_score)
        return loss

    def _freq_loss(self, block):
        if self.lambda_freq <= 0:
            return torch.zeros((), device=self.device)

        raw = self._compress_map(block["freq_raw"], mode="mix")
        enhanced = self._compress_map(block["freq_enhanced"], mode="mix")
        obj = resize_prior(block["obj_mask"], raw.shape[-2:], mode="bilinear", preserve_small=True).to(dtype=raw.dtype)
        edge = resize_prior(block["edge_mask"], raw.shape[-2:], mode="bilinear", preserve_small=True).to(dtype=raw.dtype)
        bg = resize_prior(block["bg_mask"], raw.shape[-2:], mode="bilinear", preserve_small=True).to(dtype=raw.dtype)
        cutoff = float(block.get("cutoff", 0.25))

        loss = torch.zeros((), device=raw.device)
        if self.w_hf > 0 or self.w_lf > 0:
            high_mask, low_mask = build_rfft_band_masks(raw.shape[-2], raw.shape[-1], cutoff, raw.device, raw.dtype)
            amp_raw_obj, _ = masked_rfft(raw, obj)
            amp_en_obj, _ = masked_rfft(enhanced, obj)
            amp_raw_edge, _ = masked_rfft(raw, edge)
            amp_en_edge, _ = masked_rfft(enhanced, edge)
            amp_raw_bg, _ = masked_rfft(raw, bg)
            amp_en_bg, _ = masked_rfft(enhanced, bg)

        if self.w_hf > 0:
            obj_hf_raw = (amp_raw_obj * high_mask).mean()
            obj_hf_en = (amp_en_obj * high_mask).mean()
            edge_hf_raw = (amp_raw_edge * high_mask).mean()
            edge_hf_en = (amp_en_edge * high_mask).mean()
            bg_hf_en = (amp_en_bg * high_mask).mean()
            l_hf_gain = F.relu(obj_hf_raw + self.hf_margin - obj_hf_en)
            l_hf_contrast = F.relu(self.obj_bg_margin - obj_hf_en + bg_hf_en)
            l_edge_gain = F.relu(edge_hf_raw + 0.5 * self.hf_margin - edge_hf_en)
            l_edge_contrast = F.relu(0.5 * self.obj_bg_margin - edge_hf_en + bg_hf_en)
            loss = loss + self.w_hf * (l_hf_gain + 0.5 * l_hf_contrast + 0.75 * l_edge_gain + 0.5 * l_edge_contrast)

        if self.w_lf > 0:
            bg_lf_raw = (amp_raw_bg * low_mask).mean()
            bg_lf_en = (amp_en_bg * low_mask).mean()
            l_lf = F.relu(bg_lf_en - bg_lf_raw + self.lf_margin)
            loss = loss + self.w_lf * l_lf

        if self.w_phase > 0:
            _, phase_raw_edge = masked_rfft(raw, edge)
            _, phase_en_edge = masked_rfft(enhanced, edge)
            loss = loss + self.w_phase * wrapped_phase_l1(phase_en_edge, phase_raw_edge)

        spatial_residual = self._compress_map(block.get("spatial_residual"), mode="mix")
        if spatial_residual is not None and self.w_hf > 0:
            spatial_residual = spatial_residual.abs()
            obj_s = resize_prior(obj, spatial_residual.shape[-2:], mode="bilinear", preserve_small=True).to(
                dtype=spatial_residual.dtype
            )
            edge_s = resize_prior(edge, spatial_residual.shape[-2:], mode="bilinear", preserve_small=True).to(
                dtype=spatial_residual.dtype
            )
            bg_s = resize_prior(bg, spatial_residual.shape[-2:], mode="bilinear", preserve_small=True).to(
                dtype=spatial_residual.dtype
            )
            obj_energy = self._masked_mean(spatial_residual, obj_s)
            edge_energy = self._masked_mean(spatial_residual, edge_s)
            bg_energy = self._masked_mean(spatial_residual, bg_s)
            l_spatial = F.relu(self.obj_bg_margin - obj_energy + bg_energy)
            l_spatial = l_spatial + 0.75 * F.relu(0.5 * self.obj_bg_margin - edge_energy + bg_energy)
            loss = loss + 0.5 * self.w_hf * l_spatial
        return loss

    def _task_loss(self, block):
        if self.lambda_task <= 0:
            return torch.zeros((), device=self.device)

        obj = self._compress_map(block["obj_mask"], mode="max")
        edge = self._compress_map(block.get("edge_mask"), mode="max")
        bg = self._compress_map(block["bg_mask"], mode="avg")
        prior_logits = self._compress_map(block.get("prior_logits"), mode="mix")
        gate_raw = self._compress_map(block.get("gate_raw"), mode="mix")
        gate_map = self._compress_map(block.get("gate_map"), mode="mix")
        delta_raw = self._compress_map(block.get("delta_raw"), mode="mix")
        delta_map = self._compress_map(block.get("delta_map"), mode="mix")
        patch_gate_map = self._compress_map(block.get("patch_gate_map"), mode="mix")
        patch_delta_raw = self._compress_map(block.get("patch_delta_raw"), mode="mix")
        patch_obj = self._compress_map(block.get("patch_obj_mask"), mode="max")
        patch_edge = self._compress_map(block.get("patch_edge_mask"), mode="max")
        patch_bg = self._compress_map(block.get("patch_bg_mask"), mode="avg")

        loss = torch.zeros(1, device=self.device, dtype=obj.dtype).squeeze(0)
        if prior_logits is not None:
            obj_target = resize_prior(obj, prior_logits.shape[-2:], mode="bilinear", preserve_small=True).to(
                dtype=prior_logits.dtype
            )
            pos = obj_target.sum()
            neg = obj_target.numel() - pos
            pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 20.0).to(device=prior_logits.device, dtype=prior_logits.dtype)
            prior_bce = F.binary_cross_entropy_with_logits(prior_logits, obj_target, pos_weight=pos_weight)
            prior_prob = prior_logits.sigmoid()
            intersection = (prior_prob * obj_target).sum()
            dice = 1.0 - (2.0 * intersection + 1.0) / (prior_prob.sum() + obj_target.sum() + 1.0)
            pred_edge = make_edge_mask(prior_prob, kernel_size=3).to(dtype=prior_prob.dtype)
            edge_target = resize_prior(edge, prior_logits.shape[-2:], mode="bilinear", preserve_small=True).to(
                dtype=prior_prob.dtype
            )
            edge_loss = F.l1_loss(pred_edge, edge_target)
            contrast_loss = self._map_guidance_loss(prior_prob, obj, edge, bg, target=obj_target)
            loss = loss + self.w_prior * (prior_bce + dice + 0.5 * edge_loss + 0.5 * contrast_loss)

        if gate_raw is not None:
            loss = loss + 0.25 * self.w_gate * self._map_guidance_loss(gate_raw, obj, edge, bg)
        if gate_map is not None:
            loss = loss + 0.75 * self.w_gate * self._map_guidance_loss(gate_map, obj, edge, bg)
        if delta_raw is not None:
            loss = loss + 1.25 * self.w_gate * self._map_guidance_loss(delta_raw, obj, edge, bg)
        elif delta_map is not None:
            loss = loss + 0.75 * self.w_gate * self._map_guidance_loss(delta_map, obj, edge, bg)

        if patch_gate_map is not None and patch_obj is not None and patch_edge is not None and patch_bg is not None:
            loss = loss + 0.30 * self.w_gate * self._map_guidance_loss(patch_gate_map, patch_obj, patch_edge, patch_bg)
        if patch_delta_raw is not None and patch_obj is not None and patch_edge is not None and patch_bg is not None:
            loss = loss + 0.45 * self.w_gate * self._map_guidance_loss(patch_delta_raw, patch_obj, patch_edge, patch_bg)
        return loss

    def _semantic_loss(self, block):
        """Enforce box-inside semantic compactness and object/background separability."""
        if self.lambda_sem <= 0:
            return torch.zeros((), device=self.device)

        feat = self._compress_map(block.get("semantic_feat"), mode="mix")
        if feat is None:
            return torch.zeros((), device=self.device)

        obj = resize_prior(block["obj_mask"], feat.shape[-2:], mode="bilinear", preserve_small=True).float()
        edge = resize_prior(block.get("edge_mask"), feat.shape[-2:], mode="bilinear", preserve_small=True).float()
        bg = resize_prior(block["bg_mask"], feat.shape[-2:], mode="bilinear", preserve_small=True).float()

        obj_mass = obj.flatten(1).sum(1)
        bg_mass = bg.flatten(1).sum(1)
        valid = (obj_mass > 1.0) & (bg_mass > 1.0)
        if not valid.any():
            return torch.zeros((), device=self.device)

        feat = F.normalize(feat[valid].float(), dim=1, eps=1e-6)
        obj = obj[valid]
        edge = edge[valid]
        bg = bg[valid]

        obj_proto = F.normalize(self._masked_pool(feat, obj), dim=1, eps=1e-6)
        bg_proto = F.normalize(self._masked_pool(feat, bg), dim=1, eps=1e-6)

        obj_sim = (feat * obj_proto).sum(1, keepdim=True)
        bg_sim = (feat * bg_proto).sum(1, keepdim=True)
        contrast = obj_sim - bg_sim

        target = (obj + 0.5 * edge).clamp(0.0, 1.0)
        weights = (0.25 + 2.5 * obj + 0.75 * edge + 0.25 * bg).to(dtype=contrast.dtype)
        with autocast(enabled=False):
            bce = F.binary_cross_entropy_with_logits(
                (self.sem_logit * contrast).float(), target.float(), reduction="none"
            )
        bce = (bce.to(dtype=contrast.dtype) * weights).sum() / weights.sum().clamp_min(1e-6)

        obj_gap = self._masked_mean(contrast, obj)
        edge_gap = self._masked_mean(contrast, edge)
        bg_gap = self._masked_mean(-contrast, bg)

        obj_compact = self._masked_mean(1.0 - obj_sim, obj)
        bg_compact = self._masked_mean(1.0 - bg_sim, bg)
        obj_false = self._masked_mean(F.relu(bg_sim), obj)
        bg_false = self._masked_mean(F.relu(obj_sim), bg)

        loss = 0.50 * bce
        loss = loss + 0.75 * F.relu(self.sem_margin - obj_gap)
        loss = loss + 0.75 * F.relu(self.sem_margin - bg_gap)
        loss = loss + 0.25 * F.relu(0.5 * self.sem_margin - edge_gap)
        loss = loss + 0.20 * (obj_compact + bg_compact)
        loss = loss + 0.25 * (obj_false + bg_false)
        return loss.clamp_min(0.0).to(device=self.device)

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(preds, tuple) and len(preds) == 2 and isinstance(preds[1], dict) and "blocks" in preds[1]:
            det_preds, fan_aux = preds
        else:
            det_preds, fan_aux = preds, {}
        base_total, base_items = super().__call__(det_preds, batch)
        batch_size = batch["img"].shape[0]
        self.aux_steps += 1
        aux_scale = min(self.aux_steps / self.aux_warmup_steps, 1.0)
        sem_scale = min(max((self.aux_steps - self.aux_warmup_steps) / self.aux_warmup_steps, 0.0), 1.0)

        blocks = fan_aux.get("blocks", [])
        raw_items = torch.zeros(3, device=self.device)
        opt_items = torch.zeros(3, device=self.device)
        block_count = 0
        for block in blocks:
            block_count += 1
            raw_items[0] += self._freq_loss(block)
            raw_items[1] += self._task_loss(block)
        sem_blocks = blocks[-1:] if blocks else []
        for block in sem_blocks:
            raw_items[2] += self._semantic_loss(block)

        if block_count:
            raw_items[0] = raw_items[0] / block_count
            raw_items[1] = raw_items[1] / block_count
            opt_items[0] = aux_scale * self.lambda_freq * raw_items[0]
            opt_items[1] = aux_scale * self.lambda_task * raw_items[1]
        if sem_blocks:
            raw_items[2] = raw_items[2] / len(sem_blocks)
            opt_items[2] = sem_scale * self.lambda_sem * raw_items[2]

        extra_total = opt_items * batch_size
        return torch.cat((base_total, extra_total)), torch.cat((base_items, raw_items.detach()))


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
