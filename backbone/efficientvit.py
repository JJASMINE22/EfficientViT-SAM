#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/8 16:25
# @Author  : XinYi Huang
# @FileName: efficientvit.py
# @Software: PyCharm
import torch
from torch import nn

from modules.modules import (
    ConvLayer,
    DSConv,
    MBConv,
    GLUMBConv,
    ResBlock
)
from modules.attentions import LiteMLA
from typing import List, Tuple, Optional


class EfficientViTBlock(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            num_head: Optional[int] = None,
            head_dim: int = 8,
            scales: Tuple[int] = (5,),
            expand_ratio: int = 1,
            norm=None,
            act=None,
            local_module: str = "MBConv"
    ):
        super().__init__()

        self.context_module = ResBlock(
            LiteMLA(
                in_size,
                out_size,
                num_head,
                head_dim,
                scales,
                (None, norm)
            ),
            nn.Identity()
        )

        self.local_module = ResBlock(
            eval(local_module)(
                in_size,
                out_size,
                ksize=3,
                stride=1,
                mid_size=None,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm=(None, None, norm),
                act=act
            ),
            nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        return self.local_module(self.context_module(x))


class EfficientViTBackbone(nn.Module):
    def __init__(
            self,
            widths: List,
            depths: List,
            in_channels: int = 3,
            dim: int = 32,
            expand_ratio: int = 4,
            norm: str = "batch_norm",
            act: str = "hswish"
    ):
        super().__init__()

        self.input_stem = nn.ModuleList()

        self.input_stem.append(
            ConvLayer(
                in_channels,
                widths[0],
                stride=2,
                norm=norm,
                act=act
            )
        )

        for _ in range(depths[0]):
            self.input_stem.append(
                ResBlock(
                    self.build_local_block(
                        widths[0],
                        widths[0],
                        stride=1,
                        expand_ratio=1,
                        norm=norm,
                        act=act

                    ),
                    nn.Identity()
                )
            )
        self.input_stem = nn.Sequential(*self.input_stem)
        in_channels = widths[0]

        self.stages = nn.ModuleList()
        for w, d in zip(widths[1:3], depths[1:3]):
            stage = []
            for i in range(d):
                stage.append(
                    ResBlock(
                        self.build_local_block(
                            in_channels,
                            w,
                            stride=2 if i == 0 else 1,
                            expand_ratio=expand_ratio,
                            norm=norm,
                            act=act
                        ),
                        None if i == 0 else nn.Identity()
                    )
                )
                in_channels = w
            self.stages.append(nn.Sequential(*stage))

        for w, d in zip(widths[3:], depths[3:]):
            stage = []
            stage.append(
                ResBlock(
                    self.build_local_block(
                        in_channels,
                        w,
                        stride=2,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act=act,
                        fewer_norm=True
                    ),
                    None
                )
            )
            in_channels = w

            for i in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels,
                        in_channels,
                        expand_ratio=expand_ratio,
                        head_dim=dim,
                        norm=norm,
                        act=act
                    )
                )
            self.stages.append(
                nn.Sequential(*stage)
            )

    def build_local_block(
            self,
            in_size: int,
            out_size: int,
            stride: int,
            expand_ratio: int,
            norm: str,
            act: str,
            fewer_norm: bool = False
    ):
        if expand_ratio == 1:
            block = DSConv(
                in_size,
                out_size,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act=(act, None)
            )
        else:
            block = MBConv(
                in_size,
                out_size,
                stride=stride,
                mid_size=None,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act=(act, act, None)
            )

        return block

    def forward(self, x: torch.Tensor):
        out = {}
        x = self.input_stem(x)
        out["step0"] = x
        for i, stage in enumerate(self.stages):
            x = stage(x)
            out[f"step{i + 1}"] = x

        return out
