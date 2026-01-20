#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/22 15:15
# @Author  : XinYi Huang
# @FileName: image_encoder_config.py
# @Software: PyCharm
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class ImageEncoderArgs:
    widths: list = field(
        default_factory=lambda: [32, 64, 128, 256, 512],
        metadata={
            "help": "width of each feature map"
        }
    )

    depths: list = field(
        default_factory=lambda: [1, 4, 6, 6, 9],
        metadata={
            "help": "depth of each feature map"
        }
    )

    dim: int = field(
        default=32,
        metadata={
            "help": "head dim for efficient-vit linear attn"
        }
    )

    expand_ratio: int = field(
        default=4,
        metadata={
            "help": "expand ratio for efficient-vit ffn"
        }
    )

    feature_list: list = field(
        default_factory=lambda: ["step4", "step3", "step2"],
        metadata={
            "help": "key of each feature map for neck"
        }
    )

    in_size_list: list = field(
        default_factory=lambda: [512, 256, 128],
        metadata={
            "help": "width of each feature map for neck"
        }
    )

    head_width: int = field(
        default=256,
        metadata={
            "help": "width for neck"
        }
    )

    head_depth: int = field(
        default=8,
        metadata={
            "help": "depth for neck"
        }
    )

    pretrained_dir: str = field(
        default="./hub",
        metadata={
            "help": "pretrained dir",
        }
    )


@dataclass
class PromptEncoderArgs:
    prompt_embed_dim: int = field(
        default=256,
        metadata={
            "help": "prompt embed dim"
        }
    )

    image_embedding_size: tuple = field(
        default_factory=lambda: (64, 64),
    )

    input_image_size: tuple = field(
        default_factory=lambda: (1024, 1024),
    )

    mask_in_chans: int = field(
        default=16,
        metadata={
            "help": "mask in chans"
        }
    )


@dataclass
class MaskDecoderArgs:
    depth: int = field(
        default=2,
        metadata={
            "help": "depth for mask decoder"
        }
    )

    num_heads: int = field(
        default=8,
        metadata={
            "help": "num heads for mask decoder"
        }
    )

    mlp_dim: int = field(
        default=2048,
        metadata={
            "help": "mlp dim for mask decoder"
        }
    )

    num_multimask_outputs: int = field(
        default=3,
        metadata={
            "help": "num multimask outputs"
        }
    )

    iou_head_depth: int = field(
        default=3,
        metadata={
            "help": "iou head depth"
        }
    )

    iou_head_hidden_dim: int = field(
        default=256,
        metadata={
            "help": "iou head hidden dim"
        }
    )

    attention_type: str = field(
        default="flash_attention",
        metadata={
            "help": "attention type"
        }
    )
