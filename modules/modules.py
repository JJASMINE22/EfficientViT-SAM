#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/1 17:37
# @Author  : XinYi Huang
# @FileName: modules.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import functional as F

from .utils import val2tuple, get_norm, get_act, list_sum

from typing import Optional, Tuple, Union, Dict


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            ksize: int = 3,
            stride: int = 1,
            groups: int = 1,
            use_bias: bool = False,
            dropout: float = 0.0,
            norm: Optional[Union[str, nn.Module]] = None,
            act: Optional[Union[str, nn.Module]] = None,
    ):
        super().__init__()

        self.dropout = dropout

        self.conv = nn.Conv2d(
            in_size,
            out_size,
            kernel_size=ksize,
            stride=stride,
            padding=ksize // 2,
            bias=use_bias,
            groups=groups,
        )

        self.norm = get_norm(norm, out_size)
        self.act = get_act(act)

    def forward(
            self,
            x: torch.Tensor
    ):

        x = F.dropout(x, self.dropout, self.training)

        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        return x


class DSConv(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            ksize: int = 3,
            stride: int = 1,
            use_bias=False,
            norm=None,
            act=None,
    ):
        super().__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act = val2tuple(act, 2)

        self.depth_conv = ConvLayer(
            in_size,
            in_size,
            ksize,
            stride,
            groups=in_size,
            use_bias=use_bias[0],
            norm=get_norm(norm[0], in_size),
            act=get_act(act[0])
        )

        self.point_conv = ConvLayer(
            in_size,
            out_size,
            1,
            1,
            use_bias=use_bias[1],
            norm=get_norm(norm[1], out_size),
            act=get_act(act[1])
        )

    def forward(self, x: torch.Tensor):
        x = self.depth_conv(x)
        x = self.point_conv(x)

        return x


class MBConv(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            ksize: int = 3,
            stride: int = 1,
            mid_size: Optional[int] = None,
            expand_ratio: int = 1,
            use_bias=False,
            norm=None,
            act=None,
    ):
        super().__init__()

        if mid_size is None:
            mid_size = int(in_size * expand_ratio)

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.inverted_conv = ConvLayer(
            in_size,
            mid_size,
            1,
            1,
            use_bias=use_bias[0],
            norm=get_norm(norm[0], mid_size),
            act=get_act(act[0])
        )

        self.depth_conv = ConvLayer(
            mid_size,
            mid_size,
            ksize,
            stride,
            groups=mid_size,
            use_bias=use_bias[1],
            norm=get_norm(norm[1], mid_size),
            act=get_act(act[1])
        )

        self.point_conv = ConvLayer(
            mid_size,
            out_size,
            1,
            1,
            use_bias=use_bias[2],
            norm=get_norm(norm[2], out_size),
            act=get_act(act[2])
        )

    def forward(self, x: torch.Tensor):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)

        return x


class FusedMBConv(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            ksize: int = 3,
            stride: int = 1,
            mid_size: Optional[int] = None,
            expand_ratio: int = 1,
            use_bias=False,
            norm=None,
            act=None
    ):
        super().__init__()

        if mid_size is None:
            mid_size = int(in_size * expand_ratio)

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act = val2tuple(act, 2)

        self.spatial_conv = ConvLayer(
            in_size,
            mid_size,
            ksize,
            stride,
            use_bias=use_bias[0],
            norm=get_norm(norm[0], mid_size),
            act=get_act(act[0])
        )

        self.point_conv = ConvLayer(
            mid_size,
            out_size,
            1,
            1,
            use_bias=use_bias[1],
            norm=get_norm(norm[1], out_size),
            act=get_act(act[1])
        )

    def forward(self, x: torch.Tensor):
        x = self.spatial_conv(x)
        x = self.point_conv(x)

        return x


class ResBlock(nn.Module):
    def __init__(
            self,
            main: Optional[nn.Module],
            shortcut: Optional[nn.Module],
    ):
        super().__init__()

        self.main = main
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor):
        if self.main is None:
            return x
        elif self.shortcut is None:
            return self.main(x)

        return self.shortcut(x) + self.main(x)


class GLUMBConv(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            ksize: int,
            stride: int,
            mid_size: int,
            expand_ratio: int,
            use_bias=False,
            norm=None,
            act=None
    ):
        super().__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        if mid_size is None:
            mid_size = int(in_size * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_size,
            mid_size * 2,
            ksize=1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0]
        )
        self.depth_conv = ConvLayer(
            mid_size * 2,
            mid_size * 2,
            ksize,
            stride,
            groups=mid_size * 2,
            use_bias=use_bias[1],
            norm=norm[1],
            act=act[1]
        )

        self.point_conv = ConvLayer(
            mid_size,
            out_size,
            ksize=1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2]
        )

        self.nonlinearity = nn.SiLU()

    def forward(
            self,
            x: torch.Tensor
    ):
        residual = x.clone()

        x = self.inverted_conv(x)
        x = self.nonlinearity(x)

        x = self.depth_conv(x)
        x, gate = x.chunk(2, dim=1)
        x = x * self.nonlinearity(gate)

        x = self.point_conv(x)

        if self.residual_connection:
            return residual + x

        return x


class DAGBlock(nn.Module):
    def __init__(
            self,
            inputs: Dict[str, nn.Module],
            merge: str,
            middle: nn.Module,
            outputs: Dict[str, nn.Module]
    ):
        super().__init__()

        self.input_keys = [k for k in inputs]
        self.input_ops = nn.ModuleList(list(inputs.values()))

        self.inputs = inputs
        self.merge = merge

        self.middle = middle

        self.output_keys = [k for k in outputs]
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(
            self,
            feature_dict: Dict[str, torch.Tensor]
    ):
        features = []
        for k, op in zip(self.input_keys, self.input_ops):
            features.append(op(feature_dict[k]))

        if self.merge == 'add':
            feat = list_sum(features)
        elif self.merge == 'cat':
            feat = torch.cat(features, dim=1)
        else:
            raise NotImplementedError

        feat = self.middle(feat)

        for k, op in zip(self.output_keys, self.output_ops):
            feature_dict[k] = op(feat)

        return feature_dict
