# -*- coding: utf-8 -*-

"""
@File: train_models.py
@Author: Chance (Qian Zhen)
@Description: train multiple GSCNNs based on ResNet101, ResNet152, Wide ResNet50, Wide ResNet101
@Date: 2021/12/11
"""

import argparse
from train import *
from utils import cfg, train_transform, val_transform

def train_pipeline(args):
    train_loader, val_loader = load_geoscene_train_data(cfg, train_transform, val_transform)
    for idx in range(5):
        cfg.output_num = 5
        cfg.save_path = "./ckpts/ablation/IV_ensemble/pretrain"
        cfg.model_name = "%s_pretrain_idx_%d" % (args.model_name, (idx + 1))

        model = get_model(cfg, model_name=args.model_name, pretrained=True)
        train(model, cfg, train_loader, val_loader)
        torch.cuda.empty_cache()

    train_loader, val_loader = load_train_data(cfg, train_transform, val_transform, add_confusing_data=True)
    model_weights = ["./ckpts/ablation/IV_ensemble/pretrain/%s_pretrain_idx_1_best.pth" % args.model_name,
                     "./ckpts/ablation/IV_ensemble/pretrain/%s_pretrain_idx_2_best.pth" % args.model_name,
                     "./ckpts/ablation/IV_ensemble/pretrain/%s_pretrain_idx_3_best.pth" % args.model_name,
                     "./ckpts/ablation/IV_ensemble/pretrain/%s_pretrain_idx_4_best.pth" % args.model_name,
                     "./ckpts/ablation/IV_ensemble/pretrain/%s_pretrain_idx_5_best.pth" % args.model_name]

    for idx in range(5):
        cfg.output_num = 2
        cfg.lr = 0.01
        cfg.save_path = "./ckpts/ablation/IV_ensemble/%s" % args.model_name
        cfg.model_name = "%s_finetune_idx_%d" % (args.model_name, (idx + 1))

        model = get_model(cfg, model_name=args.model_name, model_weight=model_weights[idx])
        train(model, cfg, train_loader, val_loader)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="wide_resnet50_2",
                        choices=["resnet101", "resnet152", "wide_resnet50_2", "wide_resnet101_2"])
    args = parser.parse_args()
    train_pipeline(args)