#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/1/7 12:47
# @Author  : XinYi Huang
# @FileName: __init__.py
# @Software: PyCharm
import torch

from collections import OrderedDict

from timm import create_model, list_models

if __name__ == '__main__':
    state_dict = torch.load(r"D:\datasets\sam_vit_b_01ec64.pth")

    state_dict_ = OrderedDict()
    for k, v in state_dict.items():
        if "image_encoder" in k:
            continue
        print(k, v.shape)
        state_dict_.update(
            {k: v}
        )

    # torch.save({"state_dict": state_dict_}, r"sam_vit_b_01ec64.pt")
