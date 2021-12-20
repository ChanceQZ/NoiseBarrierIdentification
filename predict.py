# -*- coding: utf-8 -*-

"""
@File: predict.py
@Author: Chance (Qian Zhen)
@Description: Prediction for BSV images in China
@Date: 2021/12/13
"""
import glob
import tqdm
import numpy as np
import torch
from models import get_model
from utils import cfg, test_transform
from dataset import load_predict_data

def model_ensemble(cfg, model_dict):
    ensemble_models = []
    for model_name, model_path in model_dict.items():
        model = get_model(cfg, model_name=model_name)
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        ensemble_models.append(model.cuda())
    return ensemble_models


@torch.no_grad()
def predict(cfg, models, data_loader):
    ensemble_pred_list, img_path_list = [], []
    for idx, (img, img_path) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        img, img_path = img.to(cfg.device), list(img_path)
        preds = []
        for model in models:
            model.eval()
            pred = model(img).cpu().numpy()
            pred = np.argmax(pred, axis=1)
            preds.append(pred)
        ensemble_pred = np.where(np.sum(preds, axis=0) >= 2, 1, 0)

        ensemble_pred_list.extend(ensemble_pred.tolist())
        img_path_list.extend(img_path)
    return ensemble_pred_list, img_path_list


if __name__ == "__main__":
    ensemble_models_dict = {
        "resnet101": "./ckpts/ablation/IV_ensemble/ensemble/resnet101/ResNet101_finetune_idx_1_best.pth",
        "resnet152": "./ckpts/ablation/IV_ensemble/ensemble/resnet152/resnet152_finetune_idx_5_best.pth",
        "wide_resnet50_2": "./ckpts/ablation/IV_ensemble/ensemble/wide_resnet50_2/wide_resnet50_2_finetune_idx_1_best.pth",
        "wide_resnet101_2": "./ckpts/ablation/IV_ensemble/ensemble/wide_resnet101_2/wide_resnet101_2_finetune_idx_2_best.pth"}
    models = model_ensemble(cfg, ensemble_models_dict)

    with open("img_paths.csv") as f:
        img_paths = f.readlines()
    img_paths = [img_path.strip() for img_path in img_paths]
    print(len(img_paths))

    data_loader = load_predict_data(cfg, img_paths, test_transform)
    ensemble_pred_list, img_path_list = predict(cfg, models, data_loader)

    with open("predict.txt", "w") as f:
        for ensemble_pred, img_path in zip(ensemble_pred_list, img_path_list):
            if ensemble_pred == 1:
                f.write(img_path)
                f.write("\n")
