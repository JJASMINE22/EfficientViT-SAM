#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/1/6 17:06
# @Author  : XinYi Huang
# @FileName: predictor.py
# @Software: PyCharm
import torch
from torch import nn

import numpy as np
from PIL import Image

from modeling.sam import EfficientViTSam
from torchvision.transforms.functional import to_tensor

from data_utils.transforms import ResizeLongestSide

from typing import Tuple, Optional


class SamPredictor(nn.Module):
    def __init__(
            self,
            sam_model: EfficientViTSam,
            transform: ResizeLongestSide,
            device: torch.device | str
    ):
        super().__init__()
        self.model = sam_model
        self.transform = transform

        self.device = device

        self.reset_image()

    def reset_image(self):
        self.input_size = None
        self.orig_size = None
        self.features = None
        self.is_image_set = False

    def set_image(self, image: Image):
        orig_image_size = (image.height, image.width)
        input_image = self.transform.apply_image(image, orig_image_size)
        input_image_torch = to_tensor(input_image).to(self.device)
        input_image_torch = input_image_torch[None]

        self.set_torch_image(input_image_torch, orig_image_size)

    def set_torch_image(
            self,
            transformed_image: torch.Tensor,
            orig_image_size: Tuple[int, int]
    ):
        self.reset_image()

        self.orig_size = orig_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
            self,
            point_coords: Optional[np.ndarray] = None,
            point_labels: Optional[np.ndarray] = None,
            box: Optional[np.ndarray] = None,
            input_mask: Optional[np.ndarray] = None,
            multimask_output: bool = True
    ):
        coords_torch, labels_torch, box_torch, mask_torch = None, None, None, None
        if point_coords is not None:
            coords_torch = torch.as_tensor(point_coords, device=self.device)
            while coords_torch.ndim < 2:
                coords_torch = coords_torch[None]
            coords_torch = self.transform.apply_coords(coords_torch, self.orig_size)
            if point_labels is None:
                labels_torch = torch.ones_like(coords_torch[:, 0])
            else:
                labels_torch = torch.as_tensor(point_labels, device=self.device)
            coords_torch = coords_torch[None]
            labels_torch = labels_torch[None]

        if box is not None:
            box_torch = torch.as_tensor(box, device=self.device)
            while box_torch.ndim < 2:
                box_torch = box_torch[None]
            box_torch = self.transform.apply_boxes(box_torch, self.orig_size)
            box_torch = box_torch[None]

        if input_mask is not None:
            mask_torch = torch.as_tensor(input_mask, device=self.device)
            mask_torch = mask_torch[None]

        masks, low_res_masks, iou_predictions = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_torch,
            multimask_output
        )

        masks_np = (masks[0].sigmoid() > .5).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
            self,
            points: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            boxes: Optional[torch.Tensor] = None,
            masks: Optional[torch.Tensor] = None,
            multimask_output: bool = True
    ):
        if not self.is_image_set:
            raise RuntimeError("An image must be set before making predictions.")

        if points is not None:
            points = (points, labels)
        else:
            points = None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points,
            boxes,
            masks
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            self.features,
            self.model.prompt_encoder.get_dense_pe(),
            sparse_embeddings,
            dense_embeddings,
            multimask_output
        )

        masks = self.model.postprocess(low_res_masks, self.input_size, self.orig_size)

        return masks, low_res_masks, iou_predictions
