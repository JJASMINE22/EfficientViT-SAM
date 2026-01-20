#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/15 17:28
# @Author  : XinYi Huang
# @FileName: sam_trainer.py
# @Software: PyCharm
import os
import random
from PIL import Image

import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from torchvision.transforms.functional import to_pil_image

from modeling.sam import (
    EfficientViTSam,
    EfficientViTBackbone,
    SamNeck,
    EfficientImageEncoder,
    PromptEncoder,
    MaskDecoder
)
from modeling.transformer import EfficientTwoWayTransformer

from modeling.utils import loss_iou, loss_masks
from data_utils.utils import masks_sample_points
from base_utils.utils import latest_checkpoint_path, load_checkpoint

import bitsandbytes as bnb


class SAMTrainer:
    mean = torch.tensor((0.485, 0.456, 0.406))
    std = torch.tensor((0.229, 0.224, 0.225))

    def __init__(
            self,
            image_encoder_args,
            prompt_encoder_args,
            mask_decoder_args,
            training_args,
            rank: int,
            steps: int
    ):
        self.training_args = training_args
        self.rank = rank
        self.step = 0

        self.efficient_sam = EfficientViTSam(
            image_encoder=EfficientImageEncoder(
                backbone=EfficientViTBackbone(
                    image_encoder_args.widths,
                    image_encoder_args.depths,
                    dim=image_encoder_args.dim,
                    expand_ratio=image_encoder_args.expand_ratio
                ),
                neck=SamNeck(
                    feature_list=image_encoder_args.feature_list,
                    in_size_list=image_encoder_args.in_size_list,
                    head_width=image_encoder_args.head_width,
                    head_depth=image_encoder_args.head_depth,
                    image_embedding_size=prompt_encoder_args.image_embedding_size
                )
            ),
            prompt_encoder=PromptEncoder(
                prompt_encoder_args.prompt_embed_dim,
                prompt_encoder_args.image_embedding_size,
                prompt_encoder_args.input_image_size,
                prompt_encoder_args.mask_in_chans
            ),
            mask_decoder=MaskDecoder(
                transformer_dim=prompt_encoder_args.prompt_embed_dim,
                transformer=EfficientTwoWayTransformer(
                    mask_decoder_args.depth,
                    prompt_encoder_args.prompt_embed_dim,
                    mask_decoder_args.num_heads,
                    mask_decoder_args.mlp_dim,
                    attention_type=mask_decoder_args.attention_type
                ),
                num_multimask_outputs=mask_decoder_args.num_multimask_outputs,
                iou_head_depth=mask_decoder_args.iou_head_depth,
                iou_head_hidden_dim=mask_decoder_args.iou_head_hidden_dim
            )
        ).cuda(rank)

        self.efficient_sam.load_state_dict(torch.load("pretrained/sam_vit_b_01ec64.pt")["state_dict"], strict=False)
        self.efficient_sam.image_encoder.load_state_dict(torch.load("pretrained/efficientvit_b3_r256.pt")["state_dict"],
                                                         strict=False)

        self.optimizer = bnb.optim.AdamW8bit(
            self.efficient_sam.parameters(),
            lr=training_args.lr,
            betas=training_args.betas,
            weight_decay=training_args.weight_decay
        )

        if training_args.resume_train:
            try:
                step = load_checkpoint(
                    latest_checkpoint_path(training_args.ckpt_dir),
                    self.efficient_sam
                )

            except Exception as e:
                raise Exception(f"received incorrect params, {repr(e)}")

            self.step = step

        self.efficient_sam = DDP(self.efficient_sam, device_ids=[rank], find_unused_parameters=True)

        self.scaler = GradScaler(enabled=training_args.enable_half)

        self.schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=steps, eta_min=training_args.min_lr
        )

        self.train_loss = 0.
        self.valid_loss = 0.

    def run_step(self, *args):
        batched_input = []
        images, masks, bboxs, points = args

        for image, mask, bbox, point in zip(images, masks, bboxs, points):
            dict_input = dict()
            dict_input["image"] = image

            if random.random() >= .5:
                dict_input["boxes"] = bbox
            else:
                try:
                    p = int(random.random() * 10 + 1)
                    dict_input["point_coords"] = masks_sample_points(mask, p)
                    dict_input["point_labels"] = torch.ones(point.shape[0], p, device=point.device)
                except Exception as e:
                    dict_input["boxes"] = bbox
            batched_input.append(
                dict_input
            )

        with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if random.random() >= .5:
                outputs, iou_outputs = self.efficient_sam(batched_input, multimask_output=True)
            else:
                outputs, iou_outputs = self.efficient_sam(batched_input, multimask_output=False)

            masks = masks.flatten(0, 1).unsqueeze(1)

            loss_list = []

            for i in range(outputs.shape[2]):
                output_i = F.interpolate(
                    outputs[:, :, i],
                    masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).reshape(-1, *masks.shape[-2:]).unsqueeze(1)

                loss_mask_i, loss_dice_i = loss_masks(
                    output_i,
                    masks,
                    num_masks=masks.shape[0]
                )

                loss_iou_i = loss_iou(
                    outputs[:, :, i].reshape(-1, *outputs.shape[-2:]).unsqueeze(1),
                    masks,
                    iou_outputs[:, :, i].flatten(),
                    num_masks=masks.shape[0]
                )

                loss_i = loss_mask_i * 20 + loss_dice_i + loss_iou_i

                loss_list.append(loss_i)
            loss = torch.stack(loss_list, -1)
            min_indices = loss.argmin(-1)

            mask = torch.zeros_like(loss, device=loss.device)
            mask.scatter_(1, min_indices.unsqueeze(-1), 1.)

            loss = (loss * mask).mean() * loss.shape[1]

        return loss

    def train(self, batch):
        images, masks, points, bboxs, _ = batch

        images = images.cuda(self.rank, non_blocking=True).contiguous()
        masks = masks.cuda(self.rank, non_blocking=True).contiguous()
        bboxs = bboxs.cuda(self.rank, non_blocking=True).contiguous()
        points = points.cuda(self.rank, non_blocking=True).contiguous()

        self.optimizer.zero_grad()
        loss = self.run_step(images, masks, bboxs, points)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if self.training_args.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.efficient_sam.parameters(), self.training_args.max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.schedular.step()

        self.train_loss += loss.item()

        self.step += 1

    @torch.no_grad()
    def validate(self, batch):

        images, masks, points, bboxs, _ = batch
        images = images.cuda(self.rank, non_blocking=True).contiguous()
        masks = masks.cuda(self.rank, non_blocking=True).contiguous()
        bboxs = bboxs.cuda(self.rank, non_blocking=True).contiguous()
        points = points.cuda(self.rank, non_blocking=True).contiguous()

        loss = self.run_step(images, masks, bboxs, points)

        self.valid_loss += loss.item()

    @torch.no_grad()
    def sample(self, batch):
        images, masks, points, bboxs, _ = batch
        random_idx = random.randint(0, images.size(0) - 1)
        image = images[random_idx].cuda(self.rank, non_blocking=True).contiguous()
        mask = masks[random_idx].cuda(self.rank, non_blocking=True).contiguous()
        bbox = bboxs[random_idx].cuda(self.rank, non_blocking=True).contiguous()
        point = points[random_idx].cuda(self.rank, non_blocking=True).contiguous()

        dict_input = dict()
        dict_input["image"] = image

        if random.random() >= .5:
            dict_input["boxes"] = bbox
        else:
            try:
                p = int(random.random() * 10 + 1)
                dict_input["point_coords"] = masks_sample_points(mask, p)
                dict_input["point_labels"] = torch.ones(point.shape[0], p, device=point.device)
            except Exception as e:
                dict_input["boxes"] = bbox

        batched_input = [dict_input]

        with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            outputs, iou_outputs = self.efficient_sam(batched_input, multimask_output=True)

        outputs = outputs.squeeze(0)
        iou_outputs = iou_outputs.squeeze(0)

        outputs = F.interpolate(
            outputs,
            image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        best_iou_idx = iou_outputs.argmax(-1, keepdim=True)

        while best_iou_idx.ndim < outputs.ndim:
            best_iou_idx = best_iou_idx[..., None]

        outputs = torch.gather(outputs, 1, best_iou_idx.expand(-1, -1, *outputs.shape[-2:]))
        mask = torch.zeros(3, *outputs.shape[-2:])
        for o in outputs.squeeze(1).cpu():
            mask += (o.sigmoid() > .7)[None].long() * torch.randint(0, 255, (3, 1, 1)) / 255.
        mask.clamp_(0., 1.)

        image = image.cpu() * self.std[:, None, None] + self.mean[:, None, None]
        image = .7 * image + .3 * mask

        to_pil_image(image).save(os.path.join(self.training_args.sample_dir, f"{self.step + 1}.png"))
