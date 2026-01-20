#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/16 15:27
# @Author  : XinYi Huang
# @FileName: utils.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    get_uncertain_point_coords_on_grid,
    point_sample
)

from typing import Tuple


def mask_iou(
        src_masks,
        tgt_masks
):
    src_masks = (src_masks > 0).float()

    intersection = (src_masks * tgt_masks).sum(dim=(-1, -2))
    union = torch.max(src_masks, tgt_masks).sum(dim=(-1, -2))

    iou = intersection / (union + 1e-6)

    return iou


def sigmoid_bce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: int,
        reduction: str = "none"
) -> torch.Tensor:
    bce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    if reduction == "none":
        return bce_loss.mean(1)

    return bce_loss.mean(1).sum() / num_masks


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: int,
        reduction: str = "none"
) -> torch.Tensor:
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)

    loss = 1 - (numerator + 1) / (denominator + 1)

    if reduction == "none":
        return loss

    return loss.sum() / num_masks


def calculate_uncertain(logits) -> torch.Tensor:
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    gt_class_logits = -torch.abs(gt_class_logits)

    return gt_class_logits


def loss_masks(
        src_masks: torch.Tensor,
        tgt_masks: torch.Tensor,
        num_masks: int,
        oversample_ratio: float = 3.0,
        reduction: str = "none",
) -> Tuple:
    with torch.no_grad():
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks.float(),
            calculate_uncertain,
            112 * 112,
            oversample_ratio,
            0.75
        )

        point_labels = point_sample(
            tgt_masks.float(),
            point_coords,
            align_corners=False
        ).squeeze(1).to(src_masks)

    point_logits = point_sample(
        src_masks.float(),
        point_coords,
        align_corners=False
    ).squeeze(1).to(src_masks)

    loss_mask = sigmoid_bce_loss(point_logits, point_labels, num_masks, reduction)
    loss_dice = dice_loss(point_logits, point_labels, num_masks, reduction)

    return loss_mask, loss_dice


def loss_iou(
        src_masks,
        tgt_masks,
        src_ious,
        num_masks,
        reduction: str = "none"
) -> torch.Tensor:
    with torch.no_grad():
        tgt_masks = F.interpolate(
            tgt_masks,
            size=src_masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        tgt_ious = mask_iou(src_masks, tgt_masks).squeeze(1)

    loss_iou = F.mse_loss(src_ious, tgt_ious, reduction="none")

    if reduction == "none":
        return loss_iou

    loss_iou = loss_iou.sum() / num_masks

    return loss_iou
