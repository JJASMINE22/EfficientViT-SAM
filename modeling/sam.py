#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/11 13:20
# @Author  : XinYi Huang
# @FileName: sam.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import functional as F
from segment_anything.modeling import PromptEncoder, MaskDecoder
from segment_anything.predictor import SamPredictor

from modules.modules import (
    ConvLayer,
    DAGBlock,
    MBConv,
    FusedMBConv,
    ResBlock
)
from modules.utils import get_norm

from backbone.efficientvit import EfficientViTBackbone

from typing import List, Optional, Dict, Any, Tuple


class SamNeck(DAGBlock):
    def __init__(
            self,
            feature_list: List[str],
            in_size_list: List[int],
            head_width: int,
            head_depth: int,
            image_embedding_size: Tuple[int],
            expand_ratio: int = 1,
            merge_op: str = "add",
            middle_op: str = "fmb",
            out_size: int = 256,
            norm: Optional[str] = "layer_norm_2d",
            act: Optional[str] = "gelu"
    ):
        inputs = {}
        for k, in_size in zip(feature_list, in_size_list):
            inputs[k] = nn.Sequential(
                ConvLayer(
                    in_size,
                    head_width,
                    ksize=1,
                    norm=norm,
                    act=None
                ),
                nn.Upsample(size=image_embedding_size, mode='bicubic', align_corners=False)
            )

        middle = []
        for i in range(head_depth):
            if middle_op == "mb":
                block = MBConv(
                    head_width,
                    head_width,
                    ksize=3,
                    stride=1,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act=(act, act, None)
                )
            elif middle_op == "fmb":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    ksize=3,
                    stride=1,
                    mid_size=head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act=(act, None)
                )
            else:
                raise NotImplementedError
            middle.append(
                ResBlock(
                    block,
                    nn.Identity()
                )
            )
        middle = nn.Sequential(*middle)

        outputs = {
            "sam_encoder": ConvLayer(
                head_width,
                out_size,
                ksize=1,
                norm=None,
                act=None
            )
        }
        super().__init__(inputs, merge_op, middle, outputs)

        self.out_size = out_size


class EfficientImageEncoder(nn.Module):
    def __init__(
            self,
            backbone: EfficientViTBackbone,
            neck: SamNeck
    ):
        super().__init__()

        self.backbone = backbone
        self.neck = neck

        self.norm = get_norm("layer_norm_2d", neck.out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        feed_dict = self.neck(feed_dict)

        output = feed_dict["sam_encoder"]
        output = self.norm(output)
        return output


class EfficientViTSam(nn.Module):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    def __init__(
            self,
            image_encoder: EfficientImageEncoder,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def forward(
            self,
            batched_input: List[Dict[str, Any]],
            multimask_output: bool = False
    ):
        images = torch.stack([x["image"] for x in batched_input], 0)

        image_embeddings = self.image_encoder(images)

        outputs = []
        iou_outputs = []
        for image_record, image_embed in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points,
                image_record.get("boxes", None),
                image_record.get("masks_inputs", None),
            )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embed.unsqueeze(0),
                self.prompt_encoder.get_dense_pe(),
                sparse_embeddings,
                dense_embeddings,
                multimask_output,
            )

            outputs.append(low_res_masks)
            iou_outputs.append(iou_predictions)

        outputs = torch.stack(outputs)
        iou_outputs = torch.stack(iou_outputs)

        return outputs, iou_outputs

    def preprocess(self, x: torch.Tensor):
        mean = self.mean.reshape(1, -1, 1, 1).to(x.device)
        std = self.std.reshape(1, -1, 1, 1).to(x.device)

        x = (x - mean) / std

        h, w = x.shape[-2:]
        pad_h = self.prompt_encoder.input_image_size[0] - h
        pad_w = self.prompt_encoder.input_image_size[1] - w

        x = F.pad(x, [0, pad_w, 0, pad_h])

        return x

    def postprocess(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...]
    ):
        masks = F.interpolate(
            masks,
            size=self.prompt_encoder.input_image_size,
            mode="bilinear",
            align_corners=False
        )

        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(
            masks,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

        return masks


if __name__ == '__main__':
    model = EfficientImageEncoder(
        backbone=EfficientViTBackbone(
            [32, 64, 128, 256, 512],
            [1, 4, 6, 6, 9]
        ),
        neck=SamNeck(
            ["step4", "step3", "step2"],
            [512, 256, 128],
            256,
            8,
            (32, 32)
        )
    ).cuda()
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    state_dict = torch.load(r"../backbone/efficientvit_b3_r256.pt")

    model.load_state_dict(state_dict, strict=True)
