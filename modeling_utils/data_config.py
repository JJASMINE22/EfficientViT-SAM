#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/24 13:16
# @Author  : XinYi Huang
# @FileName: data_config.py
# @Software: PyCharm
from dataclasses import dataclass, field


@dataclass
class DataArgs:
    """confs of training data"""

    exp_dir: str = field(
        default="./filelists", metadata={"help": "dir of image-anno paths"}
    )

    image_size: int = field(
        default=1024,
        metadata={"help": "image size"}
    )

    num_masks: int = field(
        default=4,
        metadata={"help": "num masks"}
    )

    mean: tuple = field(
        default_factory=lambda: (0.485, 0.456, 0.406),
        metadata={"help": "mean of normalization"},
    )

    std: tuple = field(
        default_factory=lambda: (0.229, 0.224, 0.225),
        metadata={"help": "std of normalization"}
    )

    seed: int = field(default=1234, metadata={"help": "random seed"})
