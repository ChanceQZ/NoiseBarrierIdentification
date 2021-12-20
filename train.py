# -*- coding: utf-8 -*-

"""
@File: train.py
@Author: Chance (Qian Zhen)
@Description: Training function
@Date: 2021/12/06
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import load_train_data, load_geoscene_train_data
from models import get_model
import warnings
warnings.filterwarnings("ignore")

def loss_fn(pred, label):
    geo_loss = nn.CrossEntropyLoss()(pred[:, :3], label[:, 0])
    nb_loss = nn.CrossEntropyLoss()(pred[:, 3:], label[:, 1])
    return geo_loss + 2 * nb_loss



@torch.no_grad()
def validation(model, cfg, criterion, val_loader):
    losses = []
    model.eval()
    for idx, (img, label) in enumerate(val_loader):
        img, label = img.to(cfg.device), label.to(cfg.device)
        label = torch.tensor(label, dtype=torch.long)
        pred = model(img)
        loss = criterion(pred, label)
        losses.append(loss.item())

    return np.array(losses).mean()


def train(model, cfg, train_loader, val_loader, fine_tune=False, **kargs):
    best_loss = 9999

    header = r'''
            Train | Valid
    Epoch |  Loss |  Loss | Time, m
    '''
    print(header)

    raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'

    model.to(cfg.device)

    if fine_tune:
        if hasattr(model, "fc"):
            output_params = list(map(id, model.fc.parameters()))
            lastlyr_params = model.fc.parameters()
        elif hasattr(model, "classifier"):
            output_params = list(map(id, model.classifier.parameters()))
            lastlyr_params = model.classifier.parameters()
        feature_params = filter(lambda p: id(p) not in output_params, model.parameters())

        optimizer = torch.optim.AdamW([
            {"params": feature_params, "lr": cfg.lr},
            {"params": lastlyr_params, "lr": cfg.lr * 10}],
                                lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs - cfg.lr_warmup_epochs
    )

    if cfg.output_num == 2:
        criterion = nn.CrossEntropyLoss()
    elif cfg.output_num == 5:
        criterion = loss_fn

    train_loss_list, val_loss_list = [], []
    for epoch in range(1, cfg.epochs + 1):
        losses = []
        start_time = time.time()
        model.train()
        for img, label in tqdm(train_loader):
            img, label = img.to(cfg.device), label.to(cfg.device)
            label = torch.tensor(label, dtype=torch.long)
            pred = model(img)

            optimizer.zero_grad()
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            lr_scheduler.step()
            losses.append(loss.item())

        val_loss = validation(model, cfg, criterion, val_loader)

        print(raw_line.format(epoch, np.array(losses).mean(), val_loss,
                              (time.time() - start_time) / 60 ** 1))

        train_loss_list.append(np.array(losses).mean())
        val_loss_list.append(val_loss)

        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.save_path, '%s_best.pth' % cfg.model_name))

    with open(os.path.join(cfg.save_path, '%s.log' % cfg.model_name), "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for idx, (train_loss, val_loss) in enumerate(zip(train_loss_list, val_loss_list)):
            f.write("%d,%.6f,%.6f\n" % (idx+1, train_loss, val_loss))


if __name__ == "__main__":
    from utils import cfg, train_transform, val_transform
    """
    # =========================Training of baseline=========================
    train_loader, val_loader = load_train_data(cfg, train_transform, val_transform)

    for idx in range(5):
        cfg.save_path = "./ckpts/ablation/I_baseline"
        cfg.model_name = "ResNet101_idx_%d" % (idx + 1)
        model = get_model(cfg, pretrained=True)
        train(model, cfg, train_loader, val_loader)
        torch.cuda.empty_cache()
    """

    # =============Training of ablation II (add geoscene prior knowledge)=============
    # Step 1: pre-training based on street view image with geoscene labels
    train_loader, val_loader = load_geoscene_train_data(cfg, train_transform, val_transform)

    for idx in range(5):
        cfg.output_num = 5
        cfg.save_path = "./ckpts/ablation/II_geoscene"
        cfg.model_name = "ResNet101_pretrain_idx_%d" % (idx + 1)

        model = get_model(cfg, pretrained=True)
        train(model, cfg, train_loader, val_loader)
        torch.cuda.empty_cache()
    # Step 2: fine-tuning based on the last step
    train_loader, val_loader = load_train_data(cfg, train_transform, val_transform)
    model_weights = ["./ckpts/ablation/II_geoscene/ResNet101_pretrain_idx_1_best.pth",
                     "./ckpts/ablation/II_geoscene/ResNet101_pretrain_idx_2_best.pth",
                     "./ckpts/ablation/II_geoscene/ResNet101_pretrain_idx_3_best.pth",
                     "./ckpts/ablation/II_geoscene/ResNet101_pretrain_idx_4_best.pth",
                     "./ckpts/ablation/II_geoscene/ResNet101_pretrain_idx_5_best.pth"]

    for idx in range(5):
        cfg.output_num = 2
        cfg.lr = 0.01
        cfg.save_path = "./ckpts/ablation/II_geoscene"
        cfg.model_name = "ResNet101_finetune_idx_%d" % (idx + 1)

        model = get_model(cfg, model_weight=model_weights[idx])
        train(model, cfg, train_loader, val_loader)
        torch.cuda.empty_cache()

    # ========Training of ablation III (add confusing negative samples' prior knowledge)========
    train_loader, val_loader = load_train_data(cfg, train_transform, val_transform, add_confusing_data=True)
    model_weights = ["./ckpts/ablation/II_geoscene/ResNet101_pretrain_idx_1_best.pth",
                     "./ckpts/ablation/II_geoscene/ResNet101_pretrain_idx_2_best.pth",
                     "./ckpts/ablation/II_geoscene/ResNet101_pretrain_idx_3_best.pth",
                     "./ckpts/ablation/II_geoscene/ResNet101_pretrain_idx_4_best.pth",
                     "./ckpts/ablation/II_geoscene/ResNet101_pretrain_idx_5_best.pth"]

    for idx in range(5):
        cfg.output_num = 2
        cfg.lr = 0.01
        cfg.save_path = "./ckpts/ablation/III_geoscene_confusing"
        cfg.model_name = "ResNet101_finetune_idx_%d" % (idx + 1)

        model = get_model(cfg, model_weight=model_weights[idx])
        train(model, cfg, train_loader, val_loader)
        torch.cuda.empty_cache()