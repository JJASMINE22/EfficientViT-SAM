#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/24 13:16
# @Author  : XinYi Huang
# @FileName: training_config.py
# @Software: PyCharm
from dataclasses import dataclass, field


@dataclass
class TrainingArgs:
    epochs: int = field(default=100, metadata={"help": "number of epochs"})
    batch_size: int = field(default=2, metadata={"help": "batch size"})
    lr: float = field(default=1e-6, metadata={"help": "learning rate"})
    lr_backbone: float = field(default=1e-7, metadata={"help": "learning rate of backbone"})
    min_lr: float = field(default=1e-7, metadata={"help": "min learning rate"})
    betas: tuple = field(default=(0.9, 0.999), metadata={"help": "betas of adam"})
    epsilon: float = field(default=1e-8, metadata={"help": "epsilon of adam"})
    weight_decay: float = field(default=0.1, metadata={"help": "weight decay"})
    momentum: float = field(default=0.9, metadata={"help": "momentum"})
    max_norm: float = field(default=1.0, metadata={"help": "max norm of the gradients"})
    accumulation_steps: int = field(
        default=1, metadata={"help": "number of accumulation steps"}
    )
    enable_half: bool = field(default=True, metadata={"help": "if using half mode"})
    port: str = field(
        default="8001", metadata={"help": "port of the distributed training"}
    )
    log_steps: int = field(default=25, metadata={"help": "logging steps"})
    gen_steps: int = field(default=25, metadata={"help": "generating steps"})
    ckpt_dir: str = field(
        default="./checkpoint", metadata={"help": "ckpt dir to save dalle"}
    )
    sample_dir: str = field(
        default="./sample", metadata={"help": "sample dir to save gen-samples"}
    )
    resume_train: bool = field(default=True, metadata={"help": "if resume training"})
    keep_ckpts: int = field(default=5, metadata={"help": "keep checkpoints"})
