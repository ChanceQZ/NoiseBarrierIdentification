# -*- coding: utf-8 -*-

"""
@File: evaluation.py
@Author: Chance (Qian Zhen)
@Description: Evaluation of model performance
@Date: 2021/12/07
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import glob
import time
import numpy as np
import torch
from dataset import load_test_data
from models import get_model
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import warnings

warnings.filterwarnings("ignore")


def eval(label_list, pred_list):
    acc = accuracy_score(label_list, pred_list)
    recall = recall_score(label_list, pred_list)
    precision = precision_score(label_list, pred_list)
    f1 = f1_score(label_list, pred_list)
    return acc, recall, precision, f1


def confidence_interval(score, mean, std):
    print("%s: %.2f%% (+/- %.2f%%)" % (score, mean * 100, std * 100))


@torch.no_grad()
def single_model_pred(cfg, model, test_loader):
    pred_list, label_list = [], []
    model.eval()
    for idx, (img, label) in enumerate(test_loader):
        img, label = img.to(cfg.device), label.cpu().numpy()

        pred = model(img).cpu().numpy()
        if pred.shape[1] == 5: pred = pred[:, -2:]
        pred = np.argmax(pred, axis=1)

        label_list.extend(label.tolist())
        pred_list.extend(pred.tolist())

    return label_list, pred_list


@torch.no_grad()
def ensemble_model_pred(cfg, ensemble_models, test_loader):
    label_list, ensemble_pred_lists = [], []
    for model in ensemble_models:
        label_list, pred_list = single_model_pred(cfg, model, test_loader)
        ensemble_pred_lists.append(pred_list)
    ensemble_pred_list = np.where(np.array(ensemble_pred_lists).sum(axis=0) >= 2, 1, 0).tolist()
    return label_list, ensemble_pred_list


def model_eval_confidence(cfg, models, test_loader, ensemble=False):
    acc_list, recall_list, precision_list, f1_list = [], [], [], []
    for model in models:
        if not ensemble:
            label_list, pred_list = single_model_pred(cfg, model, test_loader)
        else:
            ensemble_models = model
            label_list, pred_list = ensemble_model_pred(cfg, ensemble_models, test_loader)

        acc, recall, precision, f1 = eval(label_list, pred_list)
        acc_list.append(acc)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
    confidence_interval("Accuracy", np.mean(acc_list), np.std(acc_list))
    confidence_interval("Recall", np.mean(recall_list), np.std(recall_list))
    confidence_interval("Precision", np.mean(precision_list), np.std(precision_list))
    confidence_interval("F1 score", np.mean(f1_list), np.std(f1_list))


def model_capusle(cfg, model_house, ensemble=False):
    if not ensemble:
        model_paths = sorted(glob.glob(os.path.join(model_house, "*finetune*.pth")))
        models = []
        for model_path in model_paths:
            model = get_model(cfg, pretrained=True)
            model.load_state_dict(torch.load(model_path, map_location=cfg.device))
            models.append(model.cuda())

    else:
        models = []
        for idx in range(1, 6):
            model_paths = sorted(glob.glob(os.path.join(model_house, "ensemble/*/*finetune_idx_%d_best.pth" % idx)))
            ensemble_models = []
            for model_path in model_paths:
                model_name = model_path.split("\\")[-2]
                model = get_model(cfg, model_name=model_name)
                model.load_state_dict(torch.load(model_path, map_location=cfg.device))
                ensemble_models.append(model.cuda())
            models.append(ensemble_models)
    return models

def model_ensemble(cfg, model_dict):
    ensemble_models = []
    for model_name, model_path in model_dict.items():
        model = get_model(cfg, model_name=model_name)
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        ensemble_models.append(model.cuda())
    return [ensemble_models]


if __name__ == "__main__":
    from utils import cfg, test_transform

    cfg.predict_batch_size = 160
    cfg.output_num = 2

    test_loader = load_test_data(cfg, test_transform)
    # models = model_capusle(cfg, "./ckpts/ablation/IV_ensemble", ensemble=True)
    ensemble_models_dict = {"resnet101": "./ckpts/ablation/IV_ensemble/ensemble/resnet101/ResNet101_finetune_idx_1_best.pth",
                            "resnet152": "./ckpts/ablation/IV_ensemble/ensemble/resnet152/resnet152_finetune_idx_5_best.pth",
                            "wide_resnet50_2": "./ckpts/ablation/IV_ensemble/ensemble/wide_resnet50_2/wide_resnet50_2_finetune_idx_1_best.pth",
                            "wide_resnet101_2": "./ckpts/ablation/IV_ensemble/ensemble/wide_resnet101_2/wide_resnet101_2_finetune_idx_2_best.pth"}
    models = model_ensemble(cfg, ensemble_models_dict)
    model_eval_confidence(cfg, models, test_loader, ensemble=True)
