#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/12 16:37
# @Author  : XinYi Huang
# @FileName: transformer.py
# @Software: PyCharm
import torch
from torch import nn

from segment_anything.modeling.transformer import (
    TwoWayAttentionBlock,
    TwoWayTransformer
)

from segment_anything.modeling import PromptEncoder, MaskDecoder, ImageEncoderViT
from segment_anything import SamPredictor, build_sam

from modules.attentions import Attention
from typing import Type


class EfficientTwoWayAttentionBlock(TwoWayAttentionBlock):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
            skip_first_layer_pe: bool = False,
            attention_type: str = "sdpa_attention"
    ):
        super().__init__(
            embedding_dim,
            num_heads,
            mlp_dim,
            activation,
            attention_downsample_rate,
            skip_first_layer_pe,
        )

        self.self_attn = Attention(embedding_dim, num_heads, attention_type=attention_type)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, attention_type=attention_type
        )

        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, attention_type=attention_type
        )


class EfficientTwoWayTransformer(TwoWayTransformer):
    def __init__(
            self,
            depth: int,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
            attention_type: str = "sdpa_attention"
    ):
        super().__init__(
            depth,
            embedding_dim,
            num_heads,
            mlp_dim,
            activation,
            attention_downsample_rate,
        )

        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                EfficientTwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    attention_type=attention_type
                )
            )
