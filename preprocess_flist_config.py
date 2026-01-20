#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/7/2 12:03
# @Author  : XinYi Huang
# @FileName: preprocess_flist_config.py
# @Software: PyCharm
import os
import json
import random
import argparse

import multiprocessing
from tqdm import tqdm

from PIL import Image

import logging


def get_directories_recursively(data_dir: str, directories: list):
    try:
        for name in os.listdir(data_dir):
            dir = os.path.join(data_dir, name)
            if os.path.isdir(dir):
                get_directories_recursively(dir, directories)
            elif dir.endswith((".jpg", ".png", ".jpeg", ".bmp", "json")):
                directories.append(data_dir)
                break
            else:
                continue
    except Exception as e:
        logging.error(repr(e))


def get_directories(data_dir: str, dataset: str):
    directories = []

    for root, _, files in tqdm(os.walk(data_dir), desc=f"analyzing {dataset}: "):
        if files.__len__() != 0:
            if any([f.endswith((".jpg", ".png", ".jpeg", ".bmp")) for f in files]):
                directories.append(root)

    return directories


def process_single_directory(dir_path: str):
    filepaths = []
    try:
        files = os.listdir(dir_path)

        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg", ".bmp")):
                imagepath = os.path.join(dir_path, file)
                annopath = os.path.join(dir_path, file.split(".")[0] + ".json")
                try:
                    image = Image.open(imagepath)
                    image.verify()
                    if os.path.exists(annopath):
                        filepaths.append(imagepath + "\t" + annopath)
                except Exception as e:
                    logging.warning(repr(e))
                    continue

            else:
                pass

    except Exception as e:
        logging.error(repr(e))

    return filepaths


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SA1B", help="dataset")
    parser.add_argument("--exp_dir", type=str, default="./filelists", help="dir to save image paths")
    parser.add_argument("--data_dir", type=str, default=r"D:\datasets\SA1B\SA1B-A-1-1\SA1B-A-1-1",
                        help="dir to image dataset")
    args = parser.parse_args()

    # 步骤 1: 获取所有需要处理的目录列表 (在主进程完成)
    # 这通常很快，即使有几千个目录也只需要几秒
    all_dirs = get_directories(args.data_dir, args.dataset)
    print(f"扫描完成，共找到 {len(all_dirs)} 个包含图片的子目录。")

    # 步骤 2: 开启多进程处理
    process_num = 4
    results = []

    print(f"启动 {process_num} 个进程进行解析...")

    with multiprocessing.Pool(processes=process_num) as pool:
        # map 会阻塞直到所有结果返回
        # 结果是一个列表的列表：[[dir1_data...], [dir2_data...], ...]
        results_list = pool.map(process_single_directory, all_dirs)

        # 将结果展平 (Flatten)
        for item in results_list:
            results.extend(item)

    random.shuffle(results)
    train_image_mask_paths = results[:-1000]
    valid_image_mask_paths = results[-1000:]

    train_image_mask_paths = ["imagepath\tannopath"] + train_image_mask_paths
    valid_image_mask_paths = ["imagepath\tannopath"] + valid_image_mask_paths

    logging.info("Writing " + os.path.join(args.exp_dir, "train.txt").replace("\\", "/"))
    with open(os.path.join(args.exp_dir, "train.txt").replace("\\", "/"), "w", encoding="utf-8") as f:
        for info in tqdm(train_image_mask_paths, desc="writing training part: "):
            f.write(info + "\n")

    logging.info("Writing " + os.path.join(args.exp_dir, "val.txt").replace("\\", "/"))
    with open(os.path.join(args.exp_dir, "val.txt").replace("\\", "/"), "w", encoding="utf-8") as f:
        for info in tqdm(valid_image_mask_paths, desc="writing validate part: "):
            f.write(info + "\n")
