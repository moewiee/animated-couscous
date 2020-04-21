import gc
import os
import sys
import time

import apex
from apex import amp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from cvcore.configs import get_cfg_defaults
from cvcore.data import WDataset
from cvcore.modeling import EfficientNet, ResNet, DenseNet, LabelSmoothingCrossEntropyLoss, SoftmaxFocalLoss
from cvcore.solver import make_optimizer, WarmupCyclicalLR
from cvcore.utils import setup_determinism, setup_logger
from args import parse_args
from tools import train_loop, valid_model, copy_model, moving_average, bn_update, test_model


def build_model(cfg):
    if "efficientnet" in cfg.MODEL.NAME:
        model = EfficientNet
    elif "res" in cfg.MODEL.NAME:
        model = ResNet
    elif "dense" in cfg.MODEL.NAME:
        model = DenseNet
    return model(cfg)


def make_dataloader(cfg, mode, images, labels):
    dataset = WDataset(images, labels, mode=mode, cfg=cfg)
    if cfg.DEBUG:
        dataset = Subset(dataset,
                         np.random.choice(np.arange(len(dataset)), 50))
    shuffle = True if mode == "train" else False
    dataloader = DataLoader(dataset, cfg.TRAIN.BATCH_SIZE,
                            pin_memory=False, shuffle=shuffle,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader


def main(args, cfg):
    # Set logger
    logger = setup_logger(
        args.mode,
        cfg.DIRS.LOGS,
        0,
        filename=f"{cfg.EXP}.txt")

    # Declare variables
    best_metric = np.inf
    start_cycle = 0
    start_epoch = 0

    # Define model
    model = build_model(cfg)
    optimizer = make_optimizer(cfg, model)
    if args.mode == "swa":
        swa_model = build_model(cfg)

    # Define loss
    if cfg.LOSS.NAME == "ce":
        train_criterion = nn.CrossEntropyLoss()
    elif cfg.LOSS.NAME == "focal":
        train_criterion = SoftmaxFocalLoss(
            gamma=cfg.LOSS.GAMMA)
    elif cfg.LOSS.NAME == "smooth":
        train_criterion = LabelSmoothingCrossEntropyLoss(
            smoothing=0.1)

    # CUDA & Mixed Precision
    if cfg.SYSTEM.CUDA:
        if args.mode == "swa":
            model = model.cuda()
            swa_model = swa_model.cuda()
        else:
            model = model.cuda()
        train_criterion = train_criterion.cuda()

    if cfg.SYSTEM.FP16:
        bn_fp32 = True if cfg.SYSTEM.OPT_L == "O2" else None
        if args.mode == "swa":
            [model, swa_model], optimizer = amp.initialize(models=[model, swa_model], optimizers=optimizer,
                                              opt_level=cfg.SYSTEM.OPT_L,
                                              keep_batchnorm_fp32=bn_fp32)
        else:
            model, optimizer = amp.initialize(models=model, optimizers=optimizer,
                                              opt_level=cfg.SYSTEM.OPT_L,
                                              keep_batchnorm_fp32=bn_fp32)

    # Load checkpoint
    if args.load != "":
        if os.path.isfile(args.load):
            logger.info(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict(ckpt.pop('state_dict'))
            if not args.reset:
                start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
            logger.info(
                f"=> loaded checkpoint '{args.load}' (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
            if args.mode == "swa":
                ckpt = torch.load(args.load, "cpu")
                swa_model.load_state_dict(ckpt.pop('state_dict'))
        else:
            logger.info(f"=> no checkpoint found at '{args.load}'")

    # Load and split data

    if args.mode in ("train", "valid"):
        train_df = pd.read_csv(cfg.DATA.TRAIN_CSV_FILE)
        valid_df = pd.read_csv(cfg.DATA.VALIDATION_CSV_FILE)
        # train_df = df[df["fold"] != args.fold]
        # valid_df = df[df["fold"] == args.fold]

        train_loader = make_dataloader(
            cfg, "train", train_df.Image.values, train_df.Object.values)
        valid_loader = make_dataloader(
            cfg, "valid", valid_df.Image.values, valid_df.Object.values)

        scheduler = WarmupCyclicalLR("cos", cfg.OPT.BASE_LR, cfg.TRAIN.EPOCHS,
                                    iters_per_epoch=len(train_loader),
                                    warmup_epochs=cfg.OPT.WARMUP_EPOCHS)
    elif args.mode == "test":
        #TODO: Write test dataloader.
        pass


    if args.mode == "train":
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
            train_loop(logger.info, cfg, model,
                       train_loader, train_criterion, optimizer,
                       scheduler, epoch)
            _, best_metric = valid_model(logger.info, cfg, model,
                                      valid_loader, optimizer,
                                      epoch, best_metric, True)
    elif args.mode == "swa":
        delay_swa = 0
        swa_n = -1
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
            train_loop(logger.info, cfg, model,
                       train_loader, train_criterion, optimizer,
                       scheduler, epoch)
            metric, best_metric = valid_model(logger.info, cfg, model,
                                      valid_loader, optimizer,
                                      epoch, best_metric, True)
            if (epoch+1) == cfg.OPT.SWA.START:
                if metric < cfg.OPT.SWA.THRESHOLD:
                    copy_model(swa_model, model)
                    swa_n = 0
                else:
                    delay_swa = 1
            if ((epoch+1) >= cfg.OPT.SWA.START) and ((epoch+1) % cfg.OPT.SWA.FREQ == 0) and not delay_swa:
                if metric < cfg.OPT.SWA.THRESHOLD:
                    moving_average(swa_model, model, 1.0 / (swa_n + 1))
                    swa_n += 1
                    bn_update(train_loader, swa_model)
                    _, best_metric = valid_model(logger.info, cfg, swa_model,
                                            valid_loader, optimizer,
                                            epoch, best_metric, True)
                else:
                    delay_swa = 1
            if ((epoch+1) >= cfg.OPT.SWA.START) and delay_swa:
                if metric > cfg.OPT.SWA.THRESHOLD:
                    delay_swa = 0
                    if swa_n == -1:
                        copy_model(swa_model, model)
                        swa_n = 0
                    else:
                        moving_average(swa_model, model, 1.0 / (swa_n + 1))
                        swa_n += 1
                    bn_update(train_loader, swa_model)
                    _, best_metric = valid_model(logger.info, cfg, swa_model,
                                            valid_loader, optimizer,
                                            epoch, best_metric, True)



    elif args.mode == "valid":
        valid_model(logger.info, cfg, model,
                    valid_loader, optimizer, start_epoch)
    elif args.mode == "test":
        #TODO: Write test function.
        pass


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    if args.debug:
        opts = ["DEBUG", True, "TRAIN.EPOCHS", 2]
        cfg.merge_from_list(opts)

    # make dirs
    for _dir in ["WEIGHTS", "OUTPUTS", "LOGS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])
    # seed, run
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)
