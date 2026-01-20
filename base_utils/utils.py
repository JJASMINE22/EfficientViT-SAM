#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/7/11 16:04
# @Author  : XinYi Huang
# @FileName: utils.py
# @Software: PyCharm
import os
import json
import logging

import torch


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            setattr(self, k, v)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    # def __setitem__(self, key, value):
    #     return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self, index):
        return self.__dict__.get(index)


class InferHParams(HParams):
    def __init__(self, **kwargs):
        super(InferHParams, self).__init__(**kwargs)

    def __getattr__(self, index):
        return self.get(index)


def get_hparams_from_file(config_path, infer_mode=False):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config) if not infer_mode else InferHParams(**config)
    return hparams


def latest_checkpoint_path(model_dir: str):
    files = os.listdir(model_dir)
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    return os.path.join(model_dir, files[-1])


def clean_checkpoint(model_dir: str, n_ckpt_to_keep: int = 5):
    files = os.listdir(model_dir)
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if files.__len__() > n_ckpt_to_keep:
        for file in files[:-n_ckpt_to_keep]:
            os.remove(os.path.join(model_dir, file))


def load_checkpoint(ckptpath, model, optimizer=None):
    ckpt_dict = torch.load(ckptpath, map_location="cpu")

    if optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt_dict["optimizer"])
        except KeyError as e:
            logging.info(repr(e))

    if hasattr(model, "module"):
        model.module.load_state_dict(ckpt_dict["weight"], strict=False)
    else:
        model.load_state_dict(ckpt_dict["weight"], strict=False)

    logging.info("{} loaded successfully".format(ckptpath))

    try:
        step = ckpt_dict["step"]
    except KeyError as e:
        logging.info(repr(e))
        step = 0

    return step


def save_checkpoint(model, optimizer, step, ckptpath):
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {'weight': state_dict,
         'optimizer': None if optimizer is None else optimizer.state_dict(),
         'step': step}, ckptpath
    )


from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Stage(Enum):
    s1 = "TrainingStage1Args"
    s2 = "TrainingStage2Args"


@dataclass
class Trainingstate:
    stage: Optional[Stage] = None

    def set_stage(self, stage: Stage):
        self.stage = stage

    def clear_stage(self):
        self.stage = None

    def active(self, stage: Stage):
        return self.stage == stage

    def __str__(self):
        """string prompt"""
        status = {stage: self.stage == stage for stage in Stage}
        return f"TrainingStage({', '.join(f'{k.value}={v}' for k, v in status.items())})"
