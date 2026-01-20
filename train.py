#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/15 17:28
# @Author  : XinYi Huang
# @FileName: train.py
# @Software: PyCharm
import os

import logging
from tqdm import tqdm
import multiprocessing

import torch
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from sam_trainer import SAMTrainer

from data_utils.data import SAMDataset
from base_utils.utils import clean_checkpoint, save_checkpoint
from base_utils.parse_utils import CustomArgumentParser

from modeling_utils.model_config import *
from modeling_utils.training_config import TrainingArgs
from modeling_utils.data_config import DataArgs

from configure.log_config import log_config

log_config("sam_training")


def main():
    assert torch.cuda.is_available()

    parser = CustomArgumentParser(
        [
            ImageEncoderArgs,
            PromptEncoderArgs,
            MaskDecoderArgs,
            TrainingArgs,
            DataArgs
        ]
    )

    (
        image_encoder_args,
        prompt_encoder_args,
        mask_decoder_args,
        training_args,
        data_args
    ) = parser.parse_args_into_dataclasses()

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = training_args.port

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(n_gpus, image_encoder_args, prompt_encoder_args, mask_decoder_args, training_args, data_args)
    )


def run(rank, n_gpus, *args):
    (
        image_encoder_args,
        prompt_encoder_args,
        mask_decoder_args,
        training_args,
        data_args
    ) = args

    # for pytorch on win, backend use gloo
    dist.init_process_group(
        backend='gloo' if os.name == 'nt' else 'nccl', init_method='env://?use_libuv=False',
        world_size=n_gpus, rank=rank
    )
    torch.cuda.set_device(rank)

    train_dataset = SAMDataset(data_args, "train")
    valid_dataset = SAMDataset(data_args, "val")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=True)

    # when data preprocessing involves GPU inference, may throw exceptions
    num_workers = 4 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()

    train_loader = DataLoader(
        train_dataset, num_workers=0, shuffle=False, pin_memory=True,
        batch_size=training_args.batch_size, sampler=train_sampler
    )

    valid_loader = DataLoader(
        valid_dataset, num_workers=0, shuffle=False, pin_memory=True,
        batch_size=training_args.batch_size, sampler=valid_sampler
    )

    steps = training_args.epochs * train_loader.__len__()
    sam = SAMTrainer(
        image_encoder_args,
        prompt_encoder_args,
        mask_decoder_args,
        training_args,
        rank,
        steps
    )

    for epoch in range(training_args.epochs):
        train_sampler.set_epoch(epoch)

        # training part
        for batch_idx, items in enumerate(tqdm(train_loader, desc=f"Epoch{epoch + 1} train steps",
                                               total=train_loader.__len__())):
            sam.train(items)

            if rank == 0:
                if not (sam.step + 1) % training_args.gen_steps:
                    sam.sample(items)

            if not (sam.step + 1) % training_args.log_steps:
                lr = sam.optimizer.param_groups[0]["lr"]
                logging.info(
                    f'Epoch: {epoch + 1}\n'
                    f'gpu: {rank}\n'
                    f'steps: {sam.step + 1}\n'
                    f'lr: {lr}\n'
                    f'train loss is: {sam.train_loss / training_args.log_steps}\n'
                )
                # if rank == 0:
                #     save_checkpoint(
                #         sam.efficient_sam, None, sam.step + 1,
                #         os.path.join(training_args.ckpt_dir,
                #                      "sam_{}.pth.tar".format(sam.step + 1))
                #     )
                #
                #     clean_checkpoint(training_args.ckpt_dir, training_args.keep_ckpts)

                sam.train_loss = 0.

        # validate part
        # verified: switching model state during training may throw exceptions
        for batch_idx, items in enumerate(tqdm(valid_loader, desc=f"Epoch{epoch + 1} eval steps",
                                               total=valid_loader.__len__())):
            sam.validate(items)

        logging.info(
            f'gpu: {rank}\n'
            f'valid semantic loss is: {sam.valid_loss / len(valid_loader)}\n'
        )

        sam.valid_loss = 0.


if __name__ == '__main__':
    main()
