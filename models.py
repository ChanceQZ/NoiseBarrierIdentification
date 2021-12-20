# -*- coding: utf-8 -*-

"""
@File: models.py
@Author: Chance (Qian Zhen)
@Description: Model zoo
@Date: 2021/12/07
"""
import torch
from torchvision import models
import torch.nn as nn

def load_model_weight(model, model_weight, exclude_key="fc"):
    model_dict = model.state_dict()
    update_dict = {k: v for k, v in torch.load(model_weight).items() if exclude_key not in k}
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)

def get_model(cfg, model_name="resnet101", pretrained=False, model_weight=None):
    if model_name=="resnet101":
        model = models.resnet101(pretrained=pretrained)

        if model_weight is not None:
            load_model_weight(model, model_weight, "fc")

        model.fc = nn.Linear(model.fc.in_features, cfg.output_num)
        return model

    elif model_name == "resnet152":
        model = models.resnet152(pretrained=pretrained)

        if model_weight is not None:
            load_model_weight(model, model_weight, "fc")

        model.fc = nn.Linear(model.fc.in_features, cfg.output_num)
        return model

    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=pretrained)

        if model_weight is not None:
            load_model_weight(model, model_weight, "fc")

        model.fc = nn.Linear(model.fc.in_features, cfg.output_num)
        return model

    elif model_name == "wide_resnet101_2":
        model = models.wide_resnet101_2(pretrained=pretrained)

        if model_weight is not None:
            load_model_weight(model, model_weight, "fc")

        model.fc = nn.Linear(model.fc.in_features, cfg.output_num)
        return model

if __name__ == "__main__":
    from utils import cfg
    model = get_model(cfg, model_name="wide_resnet101_2")

    print(model.state_dict().keys())
    print(model.state_dict()["fc.bias"].shape)