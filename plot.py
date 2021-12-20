# -*- coding: utf-8 -*-

"""
@File: plot.py
@Author: Chance (Qian Zhen)
@Description: Plot learning curves
@Date: 2021/12/07
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for i in range(1, 6):
    log_path = "./ckpts/ablation/IV_ensemble/ensemble/wide_resnet50_2/wide_resnet50_2_finetune_idx_%d.log" % i
    log_df = pd.read_csv(log_path)
    print(np.min(log_df["val_loss"]))
# plt.figure(figsize=(10, 6))
# plt.plot(log_df["epoch"], log_df["train_loss"], label="Train")
# plt.plot(log_df["epoch"], log_df["val_loss"], label="Validation")
# plt.ylim(0, 1)
# plt.legend()
# plt.show()