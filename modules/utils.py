#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/2 15:23
# @Author  : XinYi Huang
# @FileName: utils.py
# @Software: PyCharm
import torch
from torch import nn
from typing import Any, Iterable, Union, List


class LayerNorm2d(nn.LayerNorm):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.square().mean(dim=1, keepdim=True) + self.eps).sqrt()

        if self.elementwise_affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return x


def get_norm(norm: Union[str, nn.Module], num_features: int, eps: float = 1e-6):
    if isinstance(norm, nn.Module):
        return norm
    elif isinstance(norm, str):
        if norm.lower() == "layer_norm":
            norm = nn.LayerNorm(num_features, eps=eps)
        elif norm.lower() == "layer_norm_2d":
            norm = LayerNorm2d(num_features, eps=eps)
        elif norm.lower() == "batch_norm":
            norm = nn.BatchNorm2d(num_features, eps=eps)
        else:
            raise ValueError(f"{norm} is not supported")
        return norm
    else:
        return None


def get_act(act: Union[str, nn.Module]):
    if isinstance(act, nn.Module):
        return act
    elif isinstance(act, str):
        if act.lower() == "relu":
            act = nn.ReLU()
        elif act.lower() == "gelu":
            act = nn.GELU()
        elif act.lower() == "silu":
            act = nn.SiLU()
        elif act.lower() == "hswish":
            act = nn.Hardswish()
        else:
            raise ValueError(f"{act} is not supported")

        return act
    else:
        return None


def val2tuple(val: Any, num_repeat: int):
    if not isinstance(val, Iterable) or isinstance(val, str):
        val = (val,)
    else:
        val = tuple(val)

    for i in range(num_repeat - len(val)):
        val += (val[-1],)

    return val


def list_sum(x: List[torch.Tensor]):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


if __name__ == '__main__':
    x = (1, None)
    x = val2tuple(x, 3)
    print(x)
