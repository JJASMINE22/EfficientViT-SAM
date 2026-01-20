#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/2 13:29
# @Author  : XinYi Huang
# @FileName: data.py
# @Software: PyCharm
import os
import json
from PIL import Image

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from torch.utils.data import Dataset
from data_utils.transforms import *

from typing import Any, Optional


class SAMDataset(Dataset):
    def __init__(
            self,
            args: Any,
            task: str
    ):
        super().__init__()

        assert task in ["train", "val"]

        data = pd.read_csv(
            os.path.join(
                args.exp_dir, f"{task}.txt"
            ), sep="\t"
        )
        self.data = data.sample(frac=1, random_state=args.seed)

        self.transforms = Compose(
            RandomHFlip(),
            ResizeLongestSide(
                args.image_size
            ),
            Normalize(
                mean=args.mean,
                std=args.std
            ),
            Pad(
                args.image_size
            )
        )

        self.num_masks = args.num_masks

    def process(self, image_anno_path):
        image_path, anno_path = image_anno_path

        image = Image.open(image_path).convert("RGB")
        image_size = image.size[::-1]

        annotations = json.load(
            open(anno_path, "r")
        )["annotations"]

        if len(annotations) > self.num_masks:
            r = np.random.choice(
                len(annotations), self.num_masks, replace=False
            )
        else:
            repeat, residue = self.num_masks // len(annotations), self.num_masks % len(annotations)
            r = np.concatenate(
                [np.arange(len(annotations)) for _ in range(repeat)] +
                [np.random.choice(len(annotations), residue)], axis=0
            )

        masks = np.stack([mask_utils.decode(annotations[i]['segmentation']) for i in r], 0)
        points = np.stack([annotations[i]['point_coords'][0] for i in r], 0)
        bboxs = np.stack([annotations[i]['bbox'] for i in r], 0)

        image = F.to_tensor(image).to(dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)
        points = torch.tensor(points, dtype=torch.float32)
        bboxs = torch.tensor(bboxs, dtype=torch.float32)

        bboxs[:, 2:] = bboxs[:, 2:] + bboxs[:, :2]

        args = self.transforms(
            image,
            masks,
            points,
            bboxs,
            image_size
        )

        return args

    def __getitem__(self, item) -> Optional[Tuple]:
        try:
            args = self.process(self.data.iloc[item].values)
        except Exception as e:
            print(e)
            return

        return args

    def __len__(self):
        return len(self.data)
